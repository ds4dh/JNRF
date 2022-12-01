from scr.dataloader_helper import *
from scr.loss import *
from scr.JNRF import *
import pickle
import os
import json
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# From config file.
os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open("configs/training_config.json", 'r') as openfile:
    training_config = json.load(openfile)
mydevice = torch.device(training_config["training_device"])
emb_dro = training_config["embedding_dropout"]
fnet_dro = training_config["fnet_dropout"]
hid_dim = training_config["hidden_dimension"]
fnet_num_lay = training_config["fnet_num_layer"]
fnet_exp_fac = training_config["fnet_exp_factor"]
ff_num_lay_ner = training_config["ff_num_layer_NER"]
ff_num_lay_re = training_config["ff_num_layer_RE"]
n_epoch = training_config["num_epoch"]
model_name = training_config["model_name"]

# Unpickle training/validation data
with open("data/traindatadoc", "rb") as fp:
    doc_train = pickle.load(fp)
with open("data/valdatadoc", "rb") as fp:
    doc_val = pickle.load(fp)

# Define train and val DataLoaders
train_fields = Fields(DATA=doc_train)
traindata = MyDataset(doc_train, train_fields)
trainDL = DataLoader(traindata, batch_size=1, num_workers=0, shuffle=True)
valdata = MyDataset(doc_val, train_fields)
valDL = DataLoader(valdata, batch_size=1, num_workers=0)

MyModel = JFNR(NER_dimension=len(train_fields.tagBinarizer.classes_),
                    RE_dimension=len(train_fields.relBinarizer.classes_),
                    drug_id=[2, 11],
                    entity_id=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17],
                    pretrained_embeddings_path='data/Bio_ClinicalBERT_embeddings.pt',
                    embedding_dropout=emb_dro,
                    hidden_dimension=hid_dim,
                    fnet_num_layers=fnet_num_lay,
                    fnet_dropout=fnet_dro,
                    fnet_expansion_factor=fnet_exp_fac,
                    ff_num_layers_ner=ff_num_lay_ner,
                    ff_num_layers_re=ff_num_lay_re,
                    freeze_embeddings=True)

Criterion = JointNerReLoss(10)
Optimizer = optim.Adam(MyModel.parameters(), lr=0.0003)

train_model(model=MyModel,
            train_iterator=trainDL,
            valid_iterator=valDL,
            n_epoch=n_epoch,
            optimizer=Optimizer,
            criterion=Criterion,
            device=mydevice,
            train_fields=train_fields,
            clip_value=0.00005,
            writer=True,
            gradient_accumulation_steps=1,
            save=True,
            saving_path="saved_models/"+model_name+".pt")