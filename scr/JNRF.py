from scr.postprocessing_helper import *
from scr.preprocessing_helper import Open

from copy import deepcopy
import math
import re
import os

from tqdm.auto import trange
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def gete2e(model, iterator, train_fields, device):

    znum = float(train_fields.relBinarizer.transform(
        [train_fields.no_relation_label])[0])

    for instance in iterator:
        ID = str(instance["id"].item())
        HText = instance["tok_human_text"]
        Starts = instance["starts"].numpy().squeeze()
        Ends = instance["ends"].numpy().squeeze()
        Text = instance["text"]
        Pop = instance["pop"].squeeze()
        with torch.no_grad():
            model.eval()
            NER, tempRE, (EntID, DrugID) = model(Text.to(device))
            NER, tempRE, (EntID, DrugID) = NER.cpu(
            ), tempRE.cpu(), (EntID.cpu(), DrugID.cpu())
            RE = torch.ones((NER.shape[-2], NER.shape[-2])) * float(znum)
            grid_x, grid_y = torch.meshgrid(EntID, DrugID, indexing='ij')
            try:
                RE[grid_x, grid_y] = tempRE.argmax(-1).float().unsqueeze(0)
                to_save = post_pro(NER, RE, Starts, Ends,
                                   Pop, HText, train_fields)
                write_ann(
                    to_save, ID, folder="data/loop_pred/")
            except:
                continue

    bashCommand = "python scr/evaluation_script.py data/train data/loop_pred > saved_models/loop_pred.txt"
    os.system(bashCommand)
    file = Open("saved_models/loop_pred.txt")
    row = file[-4]
    re_ = re.findall("[0-9.]+", row)
    res = {"Pre": re_[-3], "Rec": re_[-2], "F1": re_[-1]}

    return res, file[-30:-1]


class Mlp(nn.Module):

    def __init__(self, in_dim, out_dim, num_layers):

        super().__init__()

        self.num_layers = num_layers

        self.act = nn.CELU()

        self.linears = nn.ModuleList(

            [
                nn.Linear(in_dim, in_dim) for _ in range(num_layers)
            ]

        )

        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x):

        if self.num_layers != 0:
            for linear in self.linears:
                x = self.act(linear(x))
        x = self.output(x)

        return x


class FeedForward(nn.Module):
    """
    https://github.com/jaketae/fnet/tree/f2dbe626ac54a9f8528cd99cecfaee351a4a529d
    """

    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout1(self.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    """
    https://github.com/jaketae/fnet/tree/f2dbe626ac54a9f8528cd99cecfaee351a4a529d
    """
    return torch.fft.fft2(x, dim=(-1, -2)).real


class FNetEncoderLayer(nn.Module):
    """
    https://github.com/jaketae/fnet/tree/f2dbe626ac54a9f8528cd99cecfaee351a4a529d
    """

    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out


class FNet(nn.TransformerEncoder):
    """
    https://github.com/jaketae/fnet/tree/f2dbe626ac54a9f8528cd99cecfaee351a4a529d
    """

    def __init__(self, d_model=256, expansion_factor=2, dropout=0.5, num_layers=6):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_pe(hidden_dimension, num=15000):

    position = torch.arange(num).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dimension, 2)
                         * (-math.log(10000.0) / hidden_dimension))
    pe = torch.zeros(num, 1, hidden_dimension)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)

    return pe.squeeze()


class JFNR(nn.Module):

    def __init__(self,
                 NER_dimension,
                 RE_dimension,
                 drug_id,
                 entity_id,
                 pretrained_embeddings_path,
                 embedding_dropout,
                 hidden_dimension,
                 fnet_num_layers,
                 fnet_dropout,
                 fnet_expansion_factor,
                 ff_num_layers_ner,
                 ff_num_layers_re,
                 freeze_embeddings=True):

        super().__init__()

        assert isinstance(drug_id, list)
        assert isinstance(entity_id, list)
        self.drug_id = drug_id
        self.entity_id = entity_id

        # To log.
        self.NER_dimension = NER_dimension
        self.RE_dimension = RE_dimension
        self.embedding_dropout = embedding_dropout
        self.hidden_dimension = hidden_dimension
        self.fnet_num_layers = fnet_num_layers
        self.fnet_dropout = fnet_dropout
        self.fnet_expansion_factor = fnet_expansion_factor
        self.ff_num_layers_ner = ff_num_layers_ner
        self.ff_num_layers_re = ff_num_layers_re

        # Embedding.
        pretrained_embeddings = torch.load(pretrained_embeddings_path)
        vocabulary_size, embedding_dimension = pretrained_embeddings.shape
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.embedding.weight = pretrained_embeddings
        self.emb_dropout = nn.Dropout(embedding_dropout)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        self.lin1 = Mlp(embedding_dimension, hidden_dimension, 1)

        # Positional encoding.
        self.register_buffer('pe', make_pe(hidden_dimension))

        # Fourier network.
        self.fnet = FNet(
            d_model=hidden_dimension,
            expansion_factor=fnet_expansion_factor,
            dropout=fnet_dropout,
            num_layers=fnet_num_layers,
        )

        # NER.
        self.NerFF = Mlp(hidden_dimension, hidden_dimension, ff_num_layers_ner)
        self.NER = nn.Linear(hidden_dimension, NER_dimension)

        # Polynomial distance function.
        self.alpha = nn.parameter.Parameter(
            torch.tensor([[[-0.01]] for i in range(RE_dimension)]))
        self.beta = nn.parameter.Parameter(
            torch.tensor([[[0.01]] for i in range(RE_dimension)]))
        self.gamma = nn.parameter.Parameter(
            torch.tensor([[[1.]] for i in range(RE_dimension)]))

        # RE.
        self.ReFF = Mlp(hidden_dimension, hidden_dimension, ff_num_layers_re)
        self.f = nn.CELU()
        self.U = nn.Linear(hidden_dimension, hidden_dimension*RE_dimension)
        self.W = nn.Linear(hidden_dimension, hidden_dimension*RE_dimension)

    @staticmethod
    def pool_id(embs, logits, ids):
        logits = logits.argmax(dim=-1)
        pool_ids = torch.where(
            eval(" | ".join([f"(logits=={id})" for id in ids])))[0]
        return embs[:, pool_ids], pool_ids

    def forward(self, x):

        # EMBEDDING.
        len_ = x.shape[1]
        embedded = self.emb_dropout(
            self.lin1(self.embedding(x)) + self.pe[:len_, :])
        fnet_out = self.fnet(embedded)

        # NER.
        ner_in = self.NerFF(fnet_out)
        logits = self.NER(ner_in).squeeze()

        # RE.

        # RE fnet.
        re_in = self.ReFF(fnet_out)

        # Entity pooling.
        drugs, drugs_id = self.pool_id(re_in, logits, self.drug_id)
        entities, entities_id = self.pool_id(re_in, logits, self.entity_id)

        # Shapes.
        b_d, t_d, e_d = drugs.shape
        b_nod, t_nod, e_nod = entities.shape

        # Classifiers.
        UO = self.f(self.U(drugs)).reshape(b_d, t_d, e_d,
                                           self.RE_dimension).permute((0, 3, 2, 1))
        WO = self.f(self.W(entities)).reshape(b_nod, t_nod, e_nod,
                                              self.RE_dimension).permute((0, 3, 1, 2))

        # Polynomial distance function.
        dist = entities_id.reshape(-1, 1) - drugs_id
        poly_dist = ((dist**2) * self.alpha + dist *
                     self.beta + self.gamma).unsqueeze(0)

        # RE output.
        pooled_attention = WO @ UO
        pooled_attention = (pooled_attention +
                            poly_dist).permute((0, 2, 3, 1)).squeeze()

        return logits, pooled_attention, (entities_id, drugs_id)


def one_train(model,
              iterator,
              optimizer,
              criterion,
              device,
              gradient_accumulation_steps,
              writer,
              epoch,
              clip_value):

    epoch_loss = 0
    epoch_ner_loss = 0
    epoch_re_loss = 0
    grad_acc_counter = 0
    counter = 0

    model.train()

    optimizer.zero_grad()

    for c, batch in enumerate(iterator):

        # Retrieve batch elements.
        text, tag, rel = batch["text"].to(device), batch["tag"].squeeze().to(
            device), batch["rel"].squeeze().to(device)

        # Display progress.
        counter += 1
        perc = round((counter/len(iterator))*100)
        #print(f"Epoch progress : {perc}% | Current batch shape = {int(text.shape[1])}                                                                                                                                                                                                            \r", end="")

        # Compute predictions.
        NER_predictions, RE_prediction, re_ids = model(text)

        # Compute loss.
        loss, ner_loss, re_loss = criterion(NER_predictions,
                                            tag,
                                            RE_prediction,
                                            rel,
                                            re_ids)

        loss = loss / gradient_accumulation_steps
        loss.backward()
        grad_acc_counter += 1
        if grad_acc_counter % gradient_accumulation_steps == 0:
            if clip_value is not None:
                nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=clip_value)
            if writer:
                writer.add_histogram("NER.weight_batch", model.NER.weight.detach(
                ).cpu().numpy().flatten(), (epoch)*len(iterator)+(c+1))
                writer.add_scalar("NER.weight_batch_norm2", model.NER.weight.grad.data.norm(
                    2).cpu().numpy(), (epoch)*len(iterator)+(c+1))
                writer.add_histogram("NER.bias_batch", model.NER.bias.detach(
                ).cpu().numpy().flatten(), (epoch)*len(iterator)+(c+1))
                writer.add_scalar("NER.bias_batch_norm2", model.NER.bias.grad.data.norm(
                    2).cpu().numpy(), (epoch)*len(iterator)+(c+1))
            optimizer.step()
            optimizer.zero_grad()

        # Store losses.
        epoch_loss += loss.item()
        epoch_ner_loss += ner_loss
        epoch_re_loss += re_loss

    return epoch_loss/len(iterator), epoch_ner_loss/len(iterator), epoch_re_loss/len(iterator)


def evaluate(model,
             iterator,
             criterion,
             device):

    epoch_loss = 0
    epoch_ner_loss = 0
    epoch_re_loss = 0

    # Deactivating dropout layers.
    model.eval()

    # Deactivate gradient computation.
    with torch.no_grad():

        for batch in iterator:

            # Retrieve batch elements.
            text, tag, rel = batch["text"].to(device), batch["tag"].squeeze().to(
                device), batch["rel"].squeeze().to(device)

            # Compute predictions.
            NER_predictions, RE_prediction, re_ids = model(text)

            # Compute loss.
            loss, ner_loss, re_loss = criterion(NER_predictions,
                                                tag,
                                                RE_prediction,
                                                rel,
                                                re_ids)

            # Store losses.
            epoch_loss += loss.item()
            epoch_ner_loss += ner_loss
            epoch_re_loss += re_loss

    return epoch_loss/len(iterator), epoch_ner_loss/len(iterator), epoch_re_loss/len(iterator)


def train_model(model,
                train_iterator,
                valid_iterator,
                n_epoch,
                optimizer,
                criterion,
                device,
                train_fields,
                clip_value,
                writer=False,
                gradient_accumulation_steps=1,
                save=False,
                saving_path="_.pt"):

    if writer:
        if save:
            Writer = SummaryWriter(
                comment=saving_path.replace(".pt", "").replace("/", "_"))
        else:
            Writer = SummaryWriter()
        Writer.add_text(tag="NER_dimension",
                        text_string=f"{model.NER_dimension}", global_step=1)
        Writer.add_text(tag="embedding_dropout",
                        text_string=f"{model.embedding_dropout}", global_step=1)
        Writer.add_text(tag="hidden_dimension",
                        text_string=f"{model.hidden_dimension}", global_step=1)
        Writer.add_text(tag="fnet_num_layers",
                        text_string=f"{model.fnet_num_layers}", global_step=1)
        Writer.add_text(tag="fnet_dropout",
                        text_string=f"{model.fnet_dropout}", global_step=1)
        Writer.add_text(tag="fnet_expansion_factor",
                        text_string=f"{model.fnet_expansion_factor}", global_step=1)
        Writer.add_text(tag="ff_num_layers_ner",
                        text_string=f"{model.ff_num_layers_ner}", global_step=1)
        Writer.add_text(tag="ff_num_layers_re",
                        text_string=f"{model.ff_num_layers_re}", global_step=1)
    else:
        Writer = None

    # Start by assuming the best metric is negative, so that model is saved at first epoch.
    best_metric = -1

    # Put model and criterion to the desired device.
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in trange(n_epoch, desc="TRAINING"):

        train_loss, train_ner_loss, train_re_loss = one_train(model,
                                                              train_iterator,
                                                              optimizer,
                                                              criterion,
                                                              device,
                                                              gradient_accumulation_steps,
                                                              Writer,
                                                              epoch,
                                                              clip_value)

        valid_loss, val_ner_loss, val_re_loss = evaluate(model,
                                                         valid_iterator,
                                                         criterion,
                                                         device)

        val_pre_rec_f1, summary = gete2e(
            model, valid_iterator, train_fields, device)
        metric = float(val_pre_rec_f1["F1"])

        if save:
            if metric > best_metric:
                best_metric = deepcopy(metric)
                torch.save(deepcopy(model.state_dict()), saving_path)

        if writer:

            for name, param in model.named_parameters():
                if param.requires_grad:
                    Writer.add_histogram(
                        name, param.detach().cpu().numpy().flatten(), epoch+1)

            Writer.add_scalar("Train loss", train_loss, epoch+1)
            Writer.add_scalar("Train NER loss", train_ner_loss, epoch+1)
            Writer.add_scalar("Train RE loss", train_re_loss, epoch+1)

            Writer.add_scalar("Val loss", valid_loss, epoch+1)
            Writer.add_scalar("Val NER loss", val_ner_loss, epoch+1)
            Writer.add_scalar("Val RE loss", val_re_loss, epoch+1)

            Writer.add_scalar("E2E F1", metric, epoch+1)
            Writer.add_scalar("Best E2E F1", best_metric, epoch+1)

    if writer:
        Writer.close()