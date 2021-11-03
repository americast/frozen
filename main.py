import pudb
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
import torchvision
import torchvision.transforms as transforms
import torch
import data
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import os
from datetime import datetime
import torch.optim as optim

writer = SummaryWriter("logs/embedded_encoder"+str(datetime.now()).replace(" ","_"))
EPOCHS = 10000
LR = 10e-3

# Initialise the pretrained language model
tokenizer          = T5Tokenizer.from_pretrained("t5-large")
model_lang         = T5ForConditionalGeneration.from_pretrained("t5-large")
model_lang_embed   = torch.nn.Sequential(*list(model_lang.children())[:1])


# Initialise the vision model
model_vis = torchvision.models.resnet50(pretrained = True)

model_vis_ext = torch.nn.Sequential(torch.nn.Linear(1000,2048))

data_here = data.CCD(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))

dataloader = DataLoader(data_here, batch_size=160,
                        shuffle=False, num_workers=2)

loss_fct = CrossEntropyLoss(ignore_index=-100)
prev_loss = math.inf

num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

for p in model_lang.parameters():
	p.requires_grad = False

for p in model_lang_embed.parameters():
	p.requires_grad = False

model_vis = model_vis.cuda()
model_vis = torch.nn.DataParallel(model_vis)

model_vis_ext = model_vis_ext.cuda()
model_vis_ext = torch.nn.DataParallel(model_vis_ext)

model_lang_embed = model_lang_embed.cuda().eval()
model_lang_embed = torch.nn.DataParallel(model_lang_embed)

model_lang = model_lang.cuda().eval()
model_lang = torch.nn.DataParallel(model_lang)

model_vis_opt = optim.Adam(model_vis.parameters(), lr=LR)
model_vis_ext_opt = optim.Adam(model_vis_ext.parameters(), lr=LR)
for e in range(EPOCHS):
    losses = []
    for i, (img, label) in enumerate(tqdm(dataloader)):
        # Pass through vision module
        img = img.cuda()
        out_vis = model_vis(img)
        out_vis_ext = model_vis_ext(out_vis)
        out_vis_ext_1 = out_vis_ext[:, :1024].unsqueeze(1)
        out_vis_ext_2 = out_vis_ext[:, 1024:].unsqueeze(1)

        max_source_length = max_target_length = 1000

        # Pass through the embedder
        input_sequences = list(label)
        encoding = tokenizer([sequence for sequence in input_sequences],
                            padding='longest',
                            max_length=max_source_length,
                            truncation=True,
                            return_tensors="pt")
        encoding["input_ids"] = encoding["input_ids"].cuda()
        out_embedder = model_lang_embed(encoding["input_ids"])
        lang_input = torch.cat([out_vis_ext_1, out_vis_ext_2, out_embedder], axis=1)
        decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding["input_ids"])
        loss = decoder_output.loss
        loss.mean().backward()

        model_vis_opt.step()
        model_vis_ext_opt.step()
        model_vis_opt.zero_grad()
        model_vis_ext_opt.zero_grad()

        losses.append(loss.mean())
        if i % 100 == 0:
            writer.add_scalar('Loss/train', loss.sum(), e*len(dataloader) + i)

    loss_here = sum(losses)/len(losses)
    print("Epoch: "+str(e)+"; loss: "+str(loss_here))
    if loss_here < prev_loss:
        prev_loss = loss_here
        torch.save(model_vis.state_dict(), "saved_models/model_vis_"+str(e)+".pth")

