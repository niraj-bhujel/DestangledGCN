import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self,  hidden_dim=256, num_heads=4, dropout=0, batch_norm=True, residual=True, attn_dropout=0.4):

        super().__init__()

        # self.hidden_dim = hidden_dim
        # self.heads = num_heads
        # self.attn_dropout = attn_dropout
        self.batch_norm = batch_norm
        self.residual = residual


        self.sa_layer = nn.MultiheadAttention(hidden_dim,
                                                num_heads=num_heads, 
                                                dropout=attn_dropout,)

        self.ffn_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim*2, hidden_dim),
                                        nn.Dropout(dropout),
                                )

        if self.batch_norm:
            self.bn_att = nn.BatchNorm1d(hidden_dim)
            self.bn_ffn = nn.BatchNorm1d(hidden_dim)


    def forward(self, h):

        h_att, _ = self.sa_layer(h, h, h,
                        attn_mask=None,
                        key_padding_mask=None,
                        need_weights=False)

        if self.residual:
            h = h + h_att

        if self.batch_norm:
            h = self.bn_att(h)

        h = h + self.ffn_layer(h)

        if self.batch_norm:
            h = self.bn_ffn(h)

        return h
            


        