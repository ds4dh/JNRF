from scr.dataloader_helper import *
from scr.JNRF import *
import pickle
from tqdm.auto import tqdm
import json
from scr.postprocessing_helper import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))
# Unpickle train and test data
with open("data/traindatadoc", "rb") as fp:
    train = pickle.load(fp)
with open("data/testdatadoc", "rb") as fp:
    test = pickle.load(fp)
# Define test DataLoaders
train_fields = Fields(DATA=train)
testdata = MyDataset(test, train_fields)
testDL = DataLoader(testdata, batch_size=1, num_workers=0)

# From config file.
with open("configs/inference_config.json", 'r') as openfile:
    inference_config = json.load(openfile)
mydevice = torch.device(inference_config["training_device"])
emb_dro = inference_config["embedding_dropout"]
fnet_dro = inference_config["fnet_dropout"]
hid_dim = inference_config["hidden_dimension"]
fnet_num_lay = inference_config["fnet_num_layer"]
fnet_exp_fac = inference_config["fnet_exp_factor"]
ff_num_lay_ner = inference_config["ff_num_layer_NER"]
ff_num_lay_re = inference_config["ff_num_layer_RE"]
model_name = inference_config["model_name"]

# Load model.
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

MyModel.load_state_dict(torch.load("saved_models/"+model_name+".pt"))
MyModel.to(mydevice);

# Evaluate model.
c = 0
znum = 8
for instance in tqdm(testDL, desc="INFERRING"):
    ID = str(instance["id"].item())
    HText = instance["tok_human_text"]
    Starts = instance["starts"].numpy().squeeze()
    Ends = instance["ends"].numpy().squeeze()
    Text = instance["text"]
    Pop = instance["pop"].squeeze()
    with torch.no_grad():
        MyModel.eval()
        NER, tempRE, (EntID, DrugID) = MyModel(Text.to(mydevice))
        NER, tempRE, (EntID, DrugID) = NER.cpu(
        ), tempRE.cpu(), (EntID.cpu(), DrugID.cpu())
        RE = torch.ones((NER.shape[-2], NER.shape[-2])) * float(znum)
        grid_x, grid_y = torch.meshgrid(EntID, DrugID, indexing='ij')
        try:
            RE[grid_x, grid_y] = tempRE.argmax(-1).float().unsqueeze(0)
        except:
            pass
        to_save = post_pro(NER, RE, Starts, Ends, Pop, HText, train_fields)
        write_ann(to_save, ID, folder="data/predictions/")