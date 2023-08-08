import torch.nn as nn
from torchvision import models
from attention import NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_emb = v_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, return_emb=False):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)       # [batch, q_dim]
        v_emb = self.v_emb(v)           # [batch, v_dim]
        logits = self.logits(v_emb, q_emb)
        
        if return_emb:
            return logits, v_emb, q_emb
        else:
            return logits

    def logits(self, v_emb, q_emb):
        to_att = v_emb.unsqueeze(1)
        att = self.v_att(to_att, q_emb)
        v_att = (att * to_att).sum(1)   # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_att)
        joint_repr = q_repr * v_repr
        return self.classifier(joint_repr)


def build(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_emb = models.resnet18()
    v_emb.fc = nn.Identity()
    v_att = NewAttention(num_hid, num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([num_hid, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_emb, v_att, q_net, v_net, classifier)
