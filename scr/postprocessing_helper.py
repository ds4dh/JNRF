from scr.dataloader_helper import flatten
import torch
import numpy as np


def post_pro(NER, RE, Starts, Ends, Pop, HText, train_fields):

    znum = float(train_fields.relBinarizer.transform(
        [train_fields.no_relation_label])[0])

    idxdict = {}
    counter = 0
    for idx, i in enumerate(Pop):
        if i == 0:
            counter += 1
            idxdict[idx] = counter
        else:
            idxdict[idx] = counter

    output = [[[], [], [], [], []] for i in range(len(set(idxdict.values())))]

    for i in range(len(NER)):

        true_i = idxdict[i] - 1

        ner = int(NER[i].argmax(-1))
        rels = [idxdict[int(j)] - 1 for j in torch.where(RE[i] != znum)[0]]
        start = Starts[i]
        end = Ends[i]
        tok = HText[i][0]

        output[true_i][0] += [ner]
        output[true_i][1] += [start]
        output[true_i][2] += [end]
        output[true_i][3] += [tok]
        output[true_i][4] += [rels]

    for i in range(len(output)):

        output[i][1] = output[i][1][0]
        output[i][2] = output[i][2][-1]
        output[i][3] = "".join(output[i][3]).replace("##", "")
        output[i][4] = flatten(output[i][4])
        output[i][0] = output[i][0][0]

    for i in range(len(output)):

        rels = output[i][4]
        new_rels = []

        if len(rels) > 0:
            for rel in rels:
                if output[rel][0] == 2:
                    new_rels += [rel]
            new_rels = list(set(new_rels))
            if len(new_rels) > 0:
                to_keep = np.argmin(np.abs(np.array(new_rels) - i))
                new_rels = [new_rels[to_keep]]
            output[i][4] = new_rels
        else:
            continue

    Ts = {f"T{i}": output[i] for i in range(len(output))}

    for k, v in Ts.items():
        if len(v[4]) != 0:
            v[4] = list(Ts.keys())[v[4][0]]

    for k, v in Ts.copy().items():
        if v[0] == 18:
            del Ts[k]
        else:
            v[0] = train_fields.tagBinarizer.inverse_transform(np.array([v[0]]))[
                0]
            if v[0].startswith("I-"):
                del Ts[k]
            else:
                v[0] = v[0].split("-")[1]

    final_output = []
    counter = 1
    for k, v in Ts.copy().items():
        final_output.append(
            k + "\t" + v[0] + " " + str(v[1]) + " " + str(v[2]) + "\t" + v[3] + "\n")
        if len(v[4]) != 0 and v[0] != "Drug":
            final_output.append(
                f"R{counter}" + "\t" + f"{v[0]}-Drug" + " " + f"Arg1:{k}" + " " + f"Arg2:{v[4]}" + "\n")
            counter += 1

    return final_output


def write_ann(ann, ID, folder=""):

    assert isinstance(ID, str) and isinstance(folder, str)
    with open(folder + ID + '.ann', 'w', encoding='utf-8') as f:
        for item in ann:
            f.write(item)