import pudb
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
import torchvision
import torchvision.transforms as transforms
import torch
import data
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# Initialise the pretrained language model
tokenizer          = T5Tokenizer.from_pretrained("t5-large")
model_lang         = T5Model.from_pretrained("t5-large")
model_lang_encoder = T5Model.from_pretrained("t5-large").encoder
model_lang_decoder = T5Model.from_pretrained("t5-large").decoder
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

for img, label in dataloader:
    # Pass through vision module
    out_vis = model_vis(img)
    out_vis_ext = model_vis_ext(out_vis)
    out_vis_ext_1 = out_vis_ext[:, :1024].unsqueeze(1)
    out_vis_ext_2 = out_vis_ext[:, 1024:].unsqueeze(1)

    max_source_length = max_target_length = 1000

    # Pass through the encoder
    input_sequences = list(label)
    encoding = tokenizer([sequence for sequence in input_sequences],
                        padding='longest',
                        max_length=max_source_length,
                        truncation=True,
                        return_tensors="pt")
    out_encoder = model_lang_encoder(encoding["input_ids"])[0]
    decoder_input = torch.cat([out_vis_ext_1, out_vis_ext_2, out_encoder], axis=1)
    decoder_output = model_lang_decoder(decoder_input)[0]

    pu.db


    loss = loss_fct(lm_logits.view(-1, out.size(-1)), labels.view(-1))

    


    pass

# initialise CC dataset

# check how to use t5
# start training after setting zero_grad

input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids