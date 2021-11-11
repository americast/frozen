from tqdm import tqdm
import pickle
import csv
import zlib
import os

tsv_file_loc = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/Validation_GCC-Validation.tsv"
tsv_folder = "/coc/dataset/conceptual_caption/DownloadConceptualCaptions/validation/"
f = open(tsv_file_loc, "r")
read_tsv = csv.reader(f, delimiter="\t")
tsv_list = []
for row in tqdm(read_tsv):
    url = row[1]
    name = zlib.crc32(url.encode('utf-8')) & 0xffffffff
    img_path = os.path.join(tsv_folder, str(name))
    if os.path.isfile(img_path):
        tsv_list.append([row[0], row[1], name])

with open('tsv_list_validation.pkl', 'wb') as f:
   pickle.dump(tsv_list, f)