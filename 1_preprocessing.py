from scr.preprocessing_helper import *
import os
import pickle
import re
from scr.dataloader_helper import to_doc
import warnings
import json
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

# From config file.
os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open("configs/preprocessing_config.json", 'r') as openfile:
    preprocessing_config = json.load(openfile)
val_size = preprocessing_config["val_size"]
random_state = preprocessing_config["random_state"]
train_folder_path = "data/train/"
test_folder_path = "data/test/"

# Load tokenizer for preprocessing.
fast_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
my_tokenizer = fast_tokenizer.backend_tokenizer
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
torch.save(model.embeddings.word_embeddings.weight, 'data/Bio_ClinicalBERT_embeddings.pt')
# LOADING TRAIN/VAL DATA.
traintxt = [train_folder_path+f.replace("._", "") for f in os.listdir(train_folder_path) if f.endswith('.txt')]
trainann = [i.replace("txt", "ann") for i in traintxt]
X_train, X_val, y_train, y_val = train_test_split(traintxt, trainann, test_size=val_size, random_state=random_state)
# LOADING TEST DATA.
testtxt = [test_folder_path+f.replace("._", "") for f in os.listdir(test_folder_path) if f.endswith('.txt')]
testann = [i.replace("txt", "ann") for i in testtxt]
# Keep track of document IDs.
train_ids = [int(re.findall("[0-9]+", i)[0]) for i in X_train]
val_ids = [int(re.findall("[0-9]+", i)[0]) for i in X_val]
test_ids = [int(re.findall("[0-9]+", i)[0]) for i in testtxt]

# TRAIN PREPROCESSING.
print("TRAIN PREPROCESSING")
TRAINDATA = list()
for txtpath, annpath, doc_id in tqdm(list(zip(X_train, y_train, train_ids))):
    TRAINDATA.append(process_txt(doc_id, txtpath, annpath, my_tokenizer))
TRAINDATA = flatten(TRAINDATA)
TRAINDATADOC = to_doc(TRAINDATA)
with open("data/traindatadoc", "wb") as fp:
    pickle.dump(TRAINDATADOC, fp)
    
# VALIDATION PREPROCESSING.
print("VALIDATION PREPROCESSING")
VALDATA = list()
for txtpath, annpath, doc_id in tqdm(list(zip(X_val, y_val, val_ids))):
    VALDATA.append(process_txt(doc_id, txtpath, annpath, my_tokenizer))
VALDATA = flatten(VALDATA)
VALDATADOC = to_doc(VALDATA)
with open("data/valdatadoc", "wb") as fp:
    pickle.dump(VALDATADOC, fp)
    
# TEST PREPROCESSING.
print("TEST PREPROCESSING")
TESTDATA = list()
for txtpath, annpath, doc_id in tqdm(list(zip(testtxt, testann, test_ids))):
    TESTDATA.append(process_txt(doc_id, txtpath, annpath, my_tokenizer))
TESTDATA = flatten(TESTDATA)
TESTDATADOC = to_doc(TESTDATA)
with open("data/testdatadoc", "wb") as fp:
    pickle.dump(TESTDATADOC, fp)