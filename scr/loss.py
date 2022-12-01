import torch
from torch import nn


class JointNerReLoss(nn.Module):

    def __init__(self, min_entities=10):

        super().__init__()

        self.ner_loss = nn.CrossEntropyLoss()
        self.re_loss = nn.CrossEntropyLoss()
        self.min_entities = min_entities

    def forward(self, ner_input, ner_target, re_input, re_target, re_ids):

        NerLoss = self.ner_loss(ner_input, ner_target.long())
        i, j = re_ids[0], re_ids[1]

        if len(i) > self.min_entities and len(j) > self.min_entities:
            re_input = re_input.flatten(end_dim=-2)
            re_target = re_target[re_ids[0]][:, re_ids[1]].flatten()
            ReLoss = self.re_loss(re_input, re_target.long())
            return 0.5 * (NerLoss + ReLoss), NerLoss.detach().cpu().item(), ReLoss.detach().cpu().item()
        else:
            return NerLoss, NerLoss.detach().cpu().item(), 0