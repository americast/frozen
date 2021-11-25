DEBUG = False
EPOCHS = 10000
LR = 3e-4

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
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if DEBUG:
    writer = SummaryWriter("logs/DEBUG_embedded_encoder"+str(datetime.now()).replace(" ","_"))
else:
    writer = SummaryWriter("logs/embedded_encoder"+str(datetime.now()).replace(" ","_"))
# Initialise the pretrained language model
tokenizer          = T5Tokenizer.from_pretrained("t5-large")
model_lang         = T5ForConditionalGeneration.from_pretrained("t5-large")
model_lang_embed   = torch.nn.Sequential(*list(model_lang.children())[:1])


# Initialise the vision model
model_vis = torchvision.models.resnet50(pretrained = True)

model_vis_ext = torch.nn.Sequential(torch.nn.Linear(1000,2048))

dataloader = data.CCD(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), debug=DEBUG)

dataloader = DataLoader(dataloader, batch_size=128,
                        shuffle=False, num_workers=2)

dataloader_val = data.CCD(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), trainval="val")

dataloader_val = DataLoader(dataloader_val, batch_size=128,
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

model_lang_embed = model_lang_embed.cuda()
model_lang_embed = torch.nn.DataParallel(model_lang_embed)
model_lang_embed.eval()

model_lang = model_lang.cuda()
model_lang = torch.nn.DataParallel(model_lang)
model_lang.eval()

model_vis_opt = optim.Adam(model_vis.parameters(), betas=(0.9, 0.95), lr=LR)
model_vis_ext_opt = optim.Adam(model_vis_ext.parameters(), betas=(0.9, 0.95), lr=LR)

def val():
    model_vis.eval()
    model_vis_ext.eval()
    losses_val = []
    for i, (img_val, label_val) in enumerate(tqdm(dataloader_val)):
        # Pass through vision module
        img_val = img_val.cuda()
        out_vis = model_vis(img_val)
        out_vis_ext = model_vis_ext(out_vis)
        out_vis_ext_1 = out_vis_ext[:, :1024].unsqueeze(1)
        out_vis_ext_2 = out_vis_ext[:, 1024:].unsqueeze(1)

        max_source_length = max_target_length = 1000
        # Pass through the embedder
        input_sequences = list(label_val)
        encoding = tokenizer([sequence for sequence in input_sequences],
                            padding='longest',
                            max_length=max_source_length,
                            truncation=True,
                            return_tensors="pt")
        encoding["input_ids"] = encoding["input_ids"].cuda()
        encoding_pad = torch.zeros([encoding["input_ids"].shape[0],1]).int()
        encoding_final = torch.cat([encoding_pad.cuda(), encoding["input_ids"]], axis =-1)
        out_embedder = model_lang_embed(encoding_final)
        lang_input_encoder = torch.cat([out_vis_ext_1, out_vis_ext_2], axis=1)
        # encoder_outputs = model_lang.module.encoder(inputs_embeds=lang_input, return_dict=True)
        # decoder_output = model_lang.module.generate(encoder_outputs=encoder_outputs)
        # decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding["input_ids"])
        # res = tokenizer.batch_decode(decoder_output, skip_special_tokens=True)
        label_pad = torch.zeros([encoding["input_ids"].shape[0],2]).int() -100
        label_final = torch.cat([label_pad.cuda(), encoding["input_ids"]], axis = -1)
        decoder_output_for_loss = model_lang(inputs_embeds=lang_input_encoder, labels=label_final)
        loss = decoder_output_for_loss.loss
        losses_val.extend([float(x) for x in list(loss)])
        
    model_vis.train()
    model_vis_ext.train()
    return torch.exp(torch.tensor(losses_val).mean())


for e in range(EPOCHS):
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

        # Pass through the embedder
        input_sequences = list(label)
        encoding = tokenizer([sequence for sequence in input_sequences],
                            padding='longest',
                            max_length=max_source_length,
                            truncation=True,
                            return_tensors="pt")
        encoding["input_ids"] = encoding["input_ids"].cuda()
        encoding_pad = torch.zeros([encoding["input_ids"].shape[0],1]).int()
        encoding_final = torch.cat([encoding_pad.cuda(), encoding["input_ids"]], axis =-1)
        encoding_label_final = torch.cat([encoding["input_ids"], encoding_pad.cuda(),], axis =-1)
        out_embedder = model_lang_embed(encoding_final)
        lang_input_encoder = torch.cat([out_vis_ext_1, out_vis_ext_2], axis=1)
        decoder_output = model_lang(inputs_embeds=lang_input_encoder, decoder_input_ids=encoding_final, labels=encoding_label_final)
        sm = torch.nn.Softmax(dim=-1)
        poses = sm(decoder_output.logits)
        do = torch.argmax(poses, axis=-1)
        res = tokenizer.batch_decode(do, skip_special_tokens=True)
        # label_pad = torch.zeros([encoding["input_ids"].shape[0],2]).int() -100
        # label_final = torch.cat([label_pad.cuda(), encoding["input_ids"]], axis = -1)
        # if e == 100: pu.db
        loss = decoder_output.loss
        loss.mean().backward()

        model_vis_opt.step()
        model_vis_ext_opt.step()
        model_vis_opt.zero_grad()
        model_vis_ext_opt.zero_grad()

        losses.append(loss.mean())
        if i % 100 == 0:
            writer.add_scalar('Loss/train', loss.sum(), e*len(dataloader) + i)
            # sums = 0
            # for a in model_vis.module.parameters():
            #     sums += torch.sum(a)
            # for a in model_vis_ext.module.parameters():
            #     sums += torch.sum(a)
            # for a in model_lang.module.parameters():
            #     sums += torch.sum(a)
            # for a in model_lang_embed.module.parameters():
            #     sums += torch.sum(a)
            print("\nInput: "+str(encoding_final[0,:])+"\n")
            print("\nGT output: "+str(encoding_label_final[0,:])+"\n")
            print("\nPred output: "+str(do[0,:])+"\n")
            timenow = str(datetime.now()).replace(" ","_")
            print("time: "+str(timenow))
            # print("sums: "+str(sums))
            if DEBUG:
                torch.save(model_vis.state_dict(), "saved_models/DEBUG_model_vis_"+str(e)+"_"+timenow+".pth")
                torch.save(model_vis_ext.state_dict(), "saved_models/DEBUG_model_vis_ext_"+str(e)+"_"+timenow+".pth")
            else:
                torch.save(model_vis.state_dict(), "saved_models/model_vis_"+str(e)+"_"+timenow+".pth")
                torch.save(model_vis_ext.state_dict(), "saved_models/model_vis_ext_"+str(e)+"_"+timenow+".pth")
            # if DEBUG:
            #     torch.save(model_vis.state_dict(), "saved_models/DEBUG_model_vis_"+str(e)+"_"+str(i)+".pth")
            #     torch.save(model_vis_ext.state_dict(), "saved_models/DEBUG_model_vis_ext_"+str(e)+"_"+str(i)+".pth")
            # else:
            #     torch.save(model_vis.state_dict(), "saved_models/model_vis_"+str(e)+"_"+str(i)+".pth")
            #     torch.save(model_vis_ext.state_dict(), "saved_models/model_vis_ext_"+str(e)+"_"+str(i)+".pth")
            # perp = val()
            # writer.add_scalar('Perp/train', perp, e*len(dataloader) + i)
            # print("\nperp: "+str(perp)+"\n")
            loss_here = sum(losses)/len(losses)
            print("Loss: "+str(loss_here))

    loss_here = sum(losses)/len(losses)
    print("Epoch: "+str(e)+"; loss: "+str(loss_here))

