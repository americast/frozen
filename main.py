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

writer = SummaryWriter("embedded_encoder")
EPOCHS = 10000

# Initialise the pretrained language model
tokenizer          = T5Tokenizer.from_pretrained("t5-large")
model_lang         = T5ForConditionalGeneration.from_pretrained("t5-large")
model_lang_embed   = torch.nn.Sequential(*list(model_lang.children())[:1])

for p in model_lang.parameters():
	p.requires_grad = False

for p in model_lang_embed.parameters():
	p.requires_grad = False

# Initialise the vision model
model_vis = torchvision.models.resnet50(pretrained = True)

model_vis_ext = torch.nn.Sequential(torch.nn.Linear(1000,2048))

data_here = data.CCD(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))

dataloader = DataLoader(data_here, batch_size=4,
                        shuffle=True, num_workers=0)

loss_fct = CrossEntropyLoss(ignore_index=-100)
prev_loss = math.inf

for e in range(EPOCHS):
    losses = []
    for img, label in tqdm(dataloader):
        # Pass through vision module
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
        out_embedder = model_lang_embed(encoding["input_ids"])
        lang_input = torch.cat([out_vis_ext_1, out_vis_ext_2, out_embedder], axis=1)
        decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding["input_ids"])
        loss = decoder_output.loss
        loss.backward()
        losses.append(loss)

    loss_here = sum(losses)/len(losses)
    print("Epoch: "+str(e)+"; loss: "+str(loss_here))
    if loss_here < prev_loss:
        prev_loss = loss_here
        torch.save(model_vis.state_dict(), "saved_models/model_vis_"+str(e)+".pth")

    writer.add_scalar('Loss/train', sum(losses)/len(losses), e)
