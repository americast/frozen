DEBUG = True

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

# Initialise the pretrained language model
tokenizer          = T5Tokenizer.from_pretrained("t5-large")
model_lang         = T5ForConditionalGeneration.from_pretrained("t5-large")
model_lang_embed   = torch.nn.Sequential(*list(model_lang.children())[:1])


# Initialise the vision model
model_vis = torchvision.models.resnet50(pretrained = True)

model_vis_ext = torch.nn.Sequential(torch.nn.Linear(1000,2048))

if DEBUG:
    data_here = data.CCD(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), debug=DEBUG)
else:
    data_here = data.CCD(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), trainval="val")
if DEBUG:
    dataloader = DataLoader(data_here, batch_size=10,
                            shuffle=False, num_workers=2)
else:
    dataloader = DataLoader(data_here, batch_size=10,
                            shuffle=True, num_workers=2)


num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

for p in model_lang.parameters():
	p.requires_grad = False

for p in model_lang_embed.parameters():
	p.requires_grad = False

model_vis = model_vis.cuda()
model_vis = torch.nn.DataParallel(model_vis)
model_vis.load_state_dict(torch.load("saved_models/DEBUG_model_vis_1094_2021-11-28_20:01:03.614535.pth"))


model_vis_ext = model_vis_ext.cuda()
model_vis_ext = torch.nn.DataParallel(model_vis_ext)
model_vis_ext.load_state_dict(torch.load("saved_models/DEBUG_model_vis_ext_1094_2021-11-28_20:01:03.614535.pth"))

model_lang_embed = model_lang_embed.cuda()
# model_lang_embed = torch.nn.DataParallel(model_lang_embed)

model_lang = model_lang.cuda()
model_vis.eval()
model_vis_ext.eval()
model_lang_embed.eval()
model_lang.eval()
sums = 0
for a in model_vis.parameters():
    sums += torch.sum(a)
for a in model_vis_ext.parameters():
    sums += torch.sum(a)
for a in model_lang.parameters():
    sums += torch.sum(a)
for a in model_lang_embed.parameters():
    sums += torch.sum(a)
print("sums: "+str(sums))
# model_lang = torch.nn.DataParallel(model_lang)

for e in range(1):
    losses = []
    for i, (img, label) in enumerate(tqdm(dataloader)):
        # Pass through vision module
        if DEBUG:
            img = img[1:2,:,:,:]
            label = label[1:2]

        img = img.cuda()
        out_vis = model_vis(img)
        out_vis_ext = model_vis_ext(out_vis)
        out_vis_ext_1 = out_vis_ext[:, :1024].unsqueeze(1)
        out_vis_ext_2 = out_vis_ext[:, 1024:].unsqueeze(1)

        max_source_length = max_target_length = 1000
        reses = []
        input_sequences = list(label)
        encoding = tokenizer([sequence for sequence in input_sequences],
                            padding='longest',
                            max_length=max_source_length,
                            truncation=True,
                            return_tensors="pt")
        encoding["input_ids"] = encoding["input_ids"].cuda()
        encoding_pad = torch.zeros([encoding["input_ids"].shape[0],1]).int()
        encoding_final = torch.cat([encoding_pad.cuda(), encoding["input_ids"]], axis =-1)
        encoding_label_final = torch.cat([encoding_pad.cuda()*-100, encoding_pad.cuda()*-100, encoding["input_ids"], encoding_pad.cuda(),], axis =-1)
        out_embedder = model_lang_embed(encoding_final)
        out_embedder_final = torch.cat([out_vis_ext_1, out_vis_ext_2, out_embedder], axis=1)
        do_final = out_embedder_final[:,:3,:]
        do_final_tokens = torch.zeros(do_final.shape[:2]).int().cuda()

        sm = torch.nn.Softmax(dim=-1)
        for pos_here in range(3,encoding_label_final.shape[1]):
            # if pos_here == 4: pu.db
            try:
                decoder_output = model_lang(input_ids=torch.zeros(out_embedder_final.shape[:2]).int().cuda(), decoder_inputs_embeds=do_final, labels=encoding_label_final[:,:pos_here])
            except:
                pu.db
            poses = sm(decoder_output.logits)
            do = torch.argmax(poses, axis=-1)
            do_final_tokens = torch.cat([do_final_tokens, do[:,-1:]], axis=-1)
            do_embedding = model_lang_embed(do)
            do_final = torch.cat([do_final, do_embedding[:,-1:]], axis=1)

        pu.db
        res = tokenizer.batch_decode(do_final_tokens, skip_special_tokens=True)
        reses.append(res)







        # Pass through the embedder
        for bh, l in enumerate(label):

            input_sequences = list([l])
            encoding = tokenizer([sequence for sequence in input_sequences],
                                padding='longest',
                                max_length=max_source_length,
                                truncation=True,
                                return_tensors="pt")
            encoding["input_ids"] = encoding["input_ids"].cuda()
            encoding_pad = torch.zeros([encoding["input_ids"].shape[0],1]).int().cuda()
            encoding_final = torch.cat([encoding_pad.cuda(), encoding["input_ids"]], axis =-1)
            encoding_label_final = torch.cat([encoding["input_ids"], encoding_pad.cuda(),], axis =-1)
            out_embedder = model_lang_embed(encoding_final[:,0:1])
            # lang_input = torch.cat([out_vis_ext_1, out_vis_ext_2], axis=1)
            # encoder_outputs = model_lang.encoder(inputs_embeds=lang_input, return_dict=True)
            # decoder_output = model_lang.generate(encoder_outputs=encoder_outputs, decoder_input_ids=encoding_final[:,0:1])

            lang_input_encoder = torch.cat([out_vis_ext_1, out_vis_ext_2], axis=1)
            do_final = encoding_pad
            sm = torch.nn.Softmax(dim=-1)
            for pos_here in range(1,encoding_label_final.shape[1]):
                try:
                    decoder_output = model_lang(inputs_embeds=lang_input_encoder[bh,:,:].unsqueeze(0), decoder_input_ids=do_final, labels=encoding_label_final[:,:pos_here])
                except:
                    pu.db
                poses = sm(decoder_output.logits)
                do = torch.argmax(poses, axis=-1)
                do_final = torch.cat([do_final,do[:,-1:]], axis=-1)

            res = tokenizer.batch_decode(do_final, skip_special_tokens=True)
            reses.append(res)
        pass
        pu.db


        label_pad = torch.zeros([encoding["input_ids"].shape[0],2]).int() -100
        label_final = torch.cat([label_pad.cuda(), encoding["input_ids"]], axis = -1)
        decoder_output_for_loss = model_lang(inputs_embeds=lang_input, labels=label_final)
        loss = decoder_output_for_loss.loss
        pass
