import pudb
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torchvision
import torch
import data

# Initialise the pretrained language model
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model_lang = T5ForConditionalGeneration.from_pretrained("t5-large")

# Initialise the vision model
model_vis = torchvision.models.resnet50(pretrained = True)
data_here = data.CCD()
for img, label in data_here:
    pu.db
    pass

# initialise CC dataset

# check how to use t5
# start training after setting zero_grad

input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids