import pudb
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
import torchvision
import torchvision.transforms as transforms
import torch
import data
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import sys
sys.path.append("../mini-imagenet-tools")
from mini_imagenet_dataloader import MiniImageNetDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import os
from datetime import datetime
import torch.optim as optim
import numpy as np

EPOCHS = 1000
LR = 1e-5

writer = SummaryWriter("logs/embedded_encoder"+str(datetime.now()).replace(" ","_"))

# Initialise the pretrained language model
tokenizer          = T5Tokenizer.from_pretrained("t5-large")
model_lang         = T5ForConditionalGeneration.from_pretrained("t5-large")
model_lang_embed   = torch.nn.Sequential(*list(model_lang.children())[:1])


# Initialise the vision model
model_vis = torchvision.models.resnet50(pretrained = True)

model_vis_ext = torch.nn.Sequential(torch.nn.Linear(1000,2048))

dataloader = MiniImageNetDataLoader(shot_num=2, way_num=2, episode_test_sample_num=1, metatrain_folder="/srv/datasets/miniimagenet/train/", metaval_folder="/srv/datasets/miniimagenet/val/", metatest_folder="/srv/datasets/miniimagenet/test/")

# dataloader = data.miniImageNet(transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]), trainval="val")

# dataloader = DataLoader(dataloader, batch_size=30,
#                         shuffle=False, num_workers=2)


num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

for p in model_lang.parameters():
	p.requires_grad = False

for p in model_lang_embed.parameters():
	p.requires_grad = False

model_vis = model_vis.cuda()
model_vis = torch.nn.DataParallel(model_vis)
model_vis.load_state_dict(torch.load("saved_models/model_vis_0_2021-11-24_11:36:23.665906.pth"))


model_vis_ext = model_vis_ext.cuda()
model_vis_ext = torch.nn.DataParallel(model_vis_ext)
model_vis_ext.load_state_dict(torch.load("saved_models/model_vis_ext_0_2021-11-24_11:36:23.665906.pth"))


model_lang_embed = model_lang_embed.cuda()
# model_lang_embed = torch.nn.DataParallel(model_lang_embed)

model_lang = model_lang.cuda()
# model_lang = torch.nn.DataParallel(model_lang)

model_vis.eval()
model_vis_ext.eval()
model_lang_embed.eval()
model_lang.eval()

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

preds = []
preds_str = []
gts = []

for idx in tqdm(range(EPOCHS)):
    episode_train_img, episode_train_label, episode_test_img, episode_test_label = dataloader.get_batch(phase='train', idx=idx)
    episode_train_img = torch.tensor(episode_train_img).permute((0,3,1,2)).float().cuda()
    episode_test_img = torch.tensor(episode_test_img).permute((0,3,1,2)).float().cuda()
    
    # Pass through vision module
    out_vis = model_vis(episode_train_img)
    out_vis_ext = model_vis_ext(out_vis)
    out_vis_ext_1 = out_vis_ext[:, :1024].unsqueeze(1)
    out_vis_ext_2 = out_vis_ext[:, 1024:].unsqueeze(1)

    out_vis_test = model_vis(episode_test_img)
    out_vis_ext_test = model_vis_ext(out_vis_test)
    out_vis_ext_1_test = out_vis_ext_test[:, :1024].unsqueeze(1)
    out_vis_ext_2_test = out_vis_ext_test[:, 1024:].unsqueeze(1)

    max_source_length = max_target_length = 1000
    # Pass through the embedder
    induction = "Answer with dax or blicket."
    label     = []
    label_res = ["Answer with dax or blicket.", "Q: What is this? A: This is a"]
    for etl in episode_train_label:
        pos = np.argmax(etl)
        if pos == 0:
            label.append("This is a dax.")
        else:
            label.append("This is a blicket.")


    input_sequences = list(label)
    encoding = tokenizer([sequence for sequence in input_sequences],
                        padding='longest',
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt")
    encoding["input_ids"] = encoding["input_ids"].cuda()

    input_sequences_ind = [induction]
    encoding_ind = tokenizer([sequence for sequence in input_sequences_ind],
                        padding='longest',
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt")
    encoding_ind["input_ids"] = encoding_ind["input_ids"].cuda()

    input_sequences_res = list(label_res)
    encoding_res = tokenizer([sequence for sequence in input_sequences_res],
                        padding='longest',
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt")
    encoding_res["input_ids"] = encoding_res["input_ids"].cuda()
    out_embedder = model_lang_embed(encoding["input_ids"])
    out_embedder_ind = model_lang_embed(encoding_ind["input_ids"])
    out_embedder_res = model_lang_embed(encoding_res["input_ids"])
    to_cat = [out_embedder_ind[0,:,:]]
    for i in range(len(episode_train_label)):
        to_cat.append(out_vis_ext_1[i,:,:])
        to_cat.append(out_vis_ext_2[i,:,:])
        to_cat.append(out_embedder[i,:,:])
    to_cat.append(out_embedder_res[0,:,:])
    to_cat.append(out_vis_ext_1_test[0,:,:])
    to_cat.append(out_vis_ext_2_test[0,:,:])
    to_cat.append(out_embedder_res[1,:,:])


    lang_input = torch.cat(to_cat, axis=0).unsqueeze(0)
    encoder_outputs = model_lang.encoder(inputs_embeds=lang_input, return_dict=True)
    decoder_output = model_lang.generate(encoder_outputs=encoder_outputs)
    # decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding["input_ids"])
    res = tokenizer.batch_decode(decoder_output, skip_special_tokens=True)
    str_res = res[0].split()[-1].strip(".")
    if str_res[0] == "b":
        preds.append(0)
    else:
        preds.append(1)
    preds_str.append(str_res)

    pos = np.argmax(episode_test_label[0])
    gts.append(pos)
    print(res)
    """
    decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding_res["input_ids"])
    loss = decoder_output.loss
    loss.mean().backward()

    model_vis_opt.step()
    model_vis_ext_opt.step()
    model_vis_opt.zero_grad()
    model_vis_ext_opt.zero_grad()

    #Testing
    episode_test_img = torch.tensor(episode_test_img).permute((0,3,1,2)).float().cuda()
    
    # Pass through vision module
    
    out_vis = model_vis(episode_test_img)
    out_vis_ext = model_vis_ext(out_vis)
    out_vis_ext_1 = out_vis_ext[:, :1024].unsqueeze(1)
    out_vis_ext_2 = out_vis_ext[:, 1024:].unsqueeze(1)
    label_test = []
    encoder_outputs = model_lang.encoder(inputs_embeds=lang_input_test, return_dict=True)
    decoder_output = model_lang.generate(encoder_outputs=encoder_outputs)
    # decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding["input_ids"])
    res = tokenizer.batch_decode(decoder_output, skip_special_tokens=True)
    for etl in episode_test_label:
        label_test.append("This is a")

    input_sequences_test = list(label_test)
    encoding_test = tokenizer([sequence for sequence in input_sequences_test],
                        padding='longest',
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt")
    encoding_test["input_ids"] = encoding_test["input_ids"].cuda()
    out_embedder = model_lang_embed(encoding_test["input_ids"])
    lang_input_test = torch.cat([out_vis_ext_1, out_vis_ext_2, out_embedder], axis=1)
    encoder_outputs = model_lang.encoder(inputs_embeds=lang_input_test, return_dict=True)
    decoder_output = model_lang.generate(encoder_outputs=encoder_outputs)
    # decoder_output = model_lang(inputs_embeds=lang_input, labels=encoding["input_ids"])
    res = tokenizer.batch_decode(decoder_output, skip_special_tokens=True)
    pu.db
    pass
    """
pu.db