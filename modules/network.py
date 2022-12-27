import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, bert = None):
        super(Network, self).__init__()
        self.resnet = resnet
        self.bert = bert
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        extra_dim = 0 if bert is None else 768
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim + extra_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim + extra_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        x_i, x_j,
        input_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        if self.bert is not None:
            with torch.no_grad():
                self.bert.eval()
                bert_out = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )

            text_features = bert_out.pooler_output
            h_i = torch.cat([h_i, text_features], dim=-1)
            h_j = torch.cat([h_j, text_features], dim=-1)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(
        self,
        x,
        input_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ):
        h = self.resnet(x)

        if self.bert is not None:
            with torch.no_grad():
                self.bert.eval()
                bert_out = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )

            text_features = bert_out.pooler_output
            h = torch.cat([h, text_features], dim=-1)

        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
