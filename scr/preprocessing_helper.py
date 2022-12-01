import re
import numpy as np
import copy
import spacy
from pprint import pprint
nlp = spacy.load("en_core_web_sm")


def Open(path):

    with open(path) as f:

        output = f.readlines()

    return output


def sepTR(ann):

    Ts = list()
    Rs = list()
    for i in ann:

        if i[0].startswith("T"):
            Ts.append(i)
        elif i[0].startswith("R"):
            Rs.append(i)

    return Ts, Rs


def range_overlapping(x, y):

    return x.start <= y.stop-1 and y.start <= x.stop-1


def base_joint(txt):
    return "".join(txt).replace("\n", " ").replace("\t", " ")


def backprop_ws(list__):
    list_ = list__.copy()
    for i in range(len(list_)-1, 0, -1):
        np_ws = len(re.findall("(^\s*)", list_[i])[0])
        if np_ws > 0:
            list_[i] = list_[i][np_ws:]
            list_[i-1] = list_[i-1] + "".join([" "] * np_ws)
    return list_


def to_sentences(txt):
    return backprop_ws([sent.text_with_ws for sent in nlp(txt).sents])


def base_split(txt):
    output = re.split("""(\s+)""", txt)
    return list(filter(None, output))


def flatten(t):
    return [item for sublist in t for item in sublist]


def process_txt(doc_id, txtpath, annpath, tokenizer):

    txt = Open(txtpath)
    ann = Open(annpath)

    Ts, Rs = sepTR(ann)
    doc_txt = base_joint(txt)
    sent_txt = to_sentences(doc_txt)
    sent_txt_splited = [base_split(i) for i in sent_txt]

    encoded_sent = list()
    for sentence in sent_txt_splited:
        temp = list()
        for word in sentence:
            temp.append(tokenizer.encode(word.replace("#", "*")))
        encoded_sent.append(temp)

    output = list()
    ends = list()
    token_id = 0
    pop = list()
    for sent_id, encoded_sentence in enumerate(encoded_sent):
        temp = list()
        for word_id, encoded_candidates in enumerate(encoded_sentence):
            tokens = encoded_candidates.tokens[1:-1]
            ids = encoded_candidates.ids[1:-1]
            if len(tokens) != 0:
                for counter, (t, i) in enumerate(zip(tokens, ids)):
                    temp.append([doc_id, sent_id, token_id, t, i])
                    token_id += 1
                    pop.append(0 if counter == 0 else 1)
                    ends.append(len(t.replace("##", "")))
            else:
                t = sent_txt_splited[sent_id][word_id]
                i = "NOT"
                temp.append([doc_id, sent_id, np.nan, t, i])
                pop.append(0)
                ends.append(len(sent_txt_splited[sent_id][word_id]))
        output.append(temp)

    ends = np.cumsum(ends).tolist()
    starts = [0] + ends[:-1]
    ranges = list(map(range, starts, ends))

    txt1 = list()
    for i, j, k in zip(flatten(output), ranges, pop):
        if i[4] != "NOT":
            txt1.append(i[:3] + [k] + [j] + i[3:] +
                        [["O"]] + [["Z"]] + [[np.nan]])

    rs = Rs.copy()
    new_rs = list()
    for row in rs:
        row_ = row.replace("\n", "")
        temp = row_.split("\t")
        label_id, label = temp[0], temp[1]
        label, label_arg1, label_arg2 = label.split(" ")
        new_rs.append([label_arg1.replace("Arg1:", ""),
                      label, label_arg2.replace("Arg2:", "")])

    ts = Ts.copy()
    new_ts = list()
    for row in ts:
        row_ = row.replace("\n", "")
        temp_ = row_.split("\t")
        label_id, label_range, text = temp_[0], temp_[1], temp_[2]
        temp = label_range.split(" ")
        label, start, end = temp[0], temp[1], temp[-1]
        label_range_ = range(int(start), int(end))
        temp_rel = list()
        for txt_row in txt1:
            txt_row_range_ = txt_row[4]
            txt_row_token_id = txt_row[2]
            if range_overlapping(txt_row_range_, label_range_):
                temp_rel.append(txt_row_token_id)
        new_ts.append([str(label_id), str(label), temp_rel, []])

    new_ts = {i[0]: i[1:] for i in new_ts}
    for row in new_rs:
        label_arg1, label, label_arg2 = row
        new_ts[label_arg1][-1] += [new_ts[label_arg2][1]]

    labels = list(new_ts.values())

    for label in labels:
        tag = label[0]
        tag_ids = label[1]
        rel_ids = flatten(label[2])
        for c, tag_id in enumerate(tag_ids):
            txt1[tag_id][7] = ["B-"+tag if c == 0 else "I-"+tag]
            txt1[tag_id][8] = []
            txt1[tag_id][9] = []
            for rel_id in rel_ids:
                txt1[tag_id][8] += [tag+"-Drug"]
                txt1[tag_id][9] += [rel_id]

    return txt1