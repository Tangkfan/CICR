import torch
from torch import nn
import torch.nn.functional as F


class Causal_t(nn.Module):
    def __init__(self, embedding_size, num_heads=8):
        super(Causal_t, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        assert embedding_size % num_heads == 0, \
            "Embedding size must be divisible by number of heads."
        self.wy = nn.Linear(self.embedding_size, self.embedding_size)
        self.wz = nn.Linear(self.embedding_size, self.embedding_size)

        nn.init.normal_(self.wy.weight, std=0.02)
        nn.init.normal_(self.wz.weight, std=0.02)
        nn.init.constant_(self.wy.bias, 0)
        nn.init.constant_(self.wz.bias, 0)

        # self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, t, sub_dic_z, rel_dic_z, obj_dic_z):
        dic_z = torch.cat([sub_dic_z, rel_dic_z, obj_dic_z], dim=0)
        attention = torch.matmul(self.wy(t), self.wz(dic_z).t()) / (self.embedding_size ** 0.5)
        attention = F.softmax(attention, 1)
        causal_t = torch.matmul(attention, dic_z)
        # z = torch.matmul(self.prior.unsqueeze(0), z_hat).squeeze(1)  # torch.Size([box, 1, 2048])->torch.Size([box, 2048])

        return causal_t


class CounterfactualModule(nn.Module):
    def __init__(self, alpha, is_vy, fusion_mode, constant, eps=1e-8):
        super(CounterfactualModule, self).__init__()
        self.alpha = alpha
        self.is_vy = is_vy
        self.fusion_mode = fusion_mode
        self.constant = constant
        self.eps = eps

    def forward(self, z_ori, z_v, mode='train'):
        out = {}

        # both v and m are the facts
        z_vm = self.fusion(z_v, z_ori, v_fact=True, m_fact=True)  # te: total effect
        # v is the fact while m are the counterfactuals
        z_v = self.fusion(z_v, z_ori, v_fact=True, m_fact=False)  # nie: natural indirect effect

        # logits_cf = z_vm - self.alpha * z_v
        logits_cf = z_ori - self.alpha * z_v

        if mode == 'train':
            out['logits_all'] = z_vm  # for optimization
            out['logits_vq'] = z_ori  # predictions of the original model, i.e., NIE
            out['logits_cf'] = logits_cf  # predictions of CFTSG, i.e., TIE
            out['logits_v'] = z_v  # for optimization
            if self.is_vy:
                out['z_nde'] = self.fusion(z_v.clone().detach(), z_ori.clone().detach(),
                                           v_fact=True, m_fact=False)  # z_q for kl optimization with no grad
            return out
        else:
            return logits_cf

    def fusion(self, z_v, z_m, v_fact=False, m_fact=False):
        z_v, z_m = self.transform(z_v, z_m, v_fact, m_fact)

        if self.fusion_mode == 'rubi':
            z = z_m * torch.sigmoid(z_v)

        elif self.fusion_mode == 'hm':
            z = z_m * z_v
            z = torch.log(z + self.eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z = z_m + z_v
            z = torch.log(torch.sigmoid(z) + self.eps)

        return z

    def transform(self, z_v, z_m, v_fact=False, m_fact=False):
        if not m_fact:
            z_m = self.constant * torch.ones_like(z_m).cuda()

        if self.is_vy:
            if not v_fact:
                z_v = self.constant * torch.ones_like(z_v).cuda()

        if self.fusion_mode == 'hm':
            z_m = torch.sigmoid(z_m)
            if self.is_vy:
                z_v = torch.sigmoid(z_v)

        return z_v, z_m


def counterfactual_loss(prediction):
    z_q = prediction['z_nde']
    z_qkv = prediction['logits_all']
    # KL loss
    p_te = F.softmax(z_qkv, -1).clone().detach()
    p_nde = F.softmax(z_q, -1)
    kl_loss = - p_te * p_nde.log()
    cf_loss = kl_loss.sum(1).mean()
    return cf_loss
