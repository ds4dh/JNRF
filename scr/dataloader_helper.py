from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy_indexed as npi
import torch
import math


def to_doc(list_):
    temp = np.array(list_, dtype=object)
    return [i.tolist() for i in npi.group_by(temp[:, 0]).split(list_)]


def flatten(t):
    return [item for sublist in t for item in sublist]


class Fields:

    def __init__(self, DATA, no_relation_label="Z", b_drug_label="B-Drug", i_drug_label="I-Drug"):

        flattendata = np.array(flatten(DATA), dtype=object)
        self.flattendata = flattendata
        self.no_relation_label = no_relation_label
        self.b_drug_label = b_drug_label
        self.i_drug_label = i_drug_label

        self.tags = flatten(flattendata[:, 7])
        self.tagBinarizer = LabelEncoder()
        self.tagBinarizer.fit(self.tags)

        self.rel = flatten(flattendata[:, 8])
        self.relBinarizer = LabelEncoder()
        self.relBinarizer.fit(self.rel)


class MyDataset(Dataset):

    def __init__(self, DATA, train_fields):

        self.train_fields = train_fields
        self.end = list(np.cumsum([len(i) for i in DATA]))
        self.start = [0] + self.end[:-1]
        self.DATA = np.array(flatten(DATA), dtype=object)
        self.Z_num = train_fields.relBinarizer.transform(
            [train_fields.no_relation_label])[0]
        self.b_num = train_fields.tagBinarizer.transform(
            [train_fields.b_drug_label])[0]
        self.i_num = train_fields.tagBinarizer.transform(
            [train_fields.i_drug_label])[0]

    def __len__(self):

        return len(self.start)

    def __getitem__(self, item):

        # 1 - Instance
        instance = self.DATA[self.start[item]:self.end[item], :].copy()
        id = np.array(instance[0, 0], dtype=np.int32)
        reference_token_id = instance[0, 2]

        # 2 - Boundaries
        ranges = instance[:, 4].tolist()
        starts = np.array([i.start for i in ranges])
        ends = np.array([i.stop for i in ranges])

        # 3 - Human readable text
        human_text = instance[:, 5]

        # 4 - Encoded text
        text = np.array(instance[:, 6], dtype=np.int32)

        # 5 - Tags
        tags = flatten(instance[:, 7].tolist())
        tag = np.array(self.train_fields.tagBinarizer.transform(
            tags), dtype=np.int32)

        # 6 - Relations
        relcoltypes = instance[:, 8]
        relrows = instance[:, 9]

        dim = len(instance)
        targetrel = np.ones((dim, dim), dtype=np.int32) * self.Z_num
        for counter, relrow in enumerate(relrows):
            if len(relrow) > 0:
                if math.isnan(relrow[0]):
                    continue
                else:
                    reltypes = self.train_fields.relBinarizer.transform(
                        relcoltypes[counter])
                    for uniquerelcol, uniquereltype in zip(relrow, reltypes):
                        try:
                            cand_tag = tag[uniquerelcol - reference_token_id]
                            if cand_tag in [self.b_num, self.i_num]:
                                targetrel[counter, uniquerelcol -
                                          reference_token_id] = uniquereltype
                            else:
                                pass
                        except:
                            # Out of range relations. Can happen often in sentence level modeling.
                            pass

        # part_of_previous embedding.
        pop = np.array(instance[:, 3], dtype=np.int32)

        return {
            "id": id,
            "starts": starts,
            "ends": ends,
            "pop": pop,
            "tok_human_text": list(human_text),
            "text": text,
            "tag": tag,
            "rel": targetrel
        }