import torch.nn.functional as F
import torch.nn as nn

from torch.nn.modules import Embedding
from .layernorm import LILayerNorm
from .attention import LIMultiheadAttention
from .linear import LILinear

import copy


class LIEmbedding(Embedding):
    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
        super(LIEmbedding, self).__init__(num_embeddings, embedding_dim,
                                          padding_idx=padding_idx, max_norm=max_norm,
                                          norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                          sparse=sparse)

    def interpolated_forward(self, x, alpha, w1, w2):
        w = (1 - alpha) * w1 + alpha * w2
        return F.embedding(
            x, w, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class LITransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(LITransformerEncoderLayer, self).__init__()
        self.self_attn = LIMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = LILinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LILinear(dim_feedforward, d_model)

        self.norm1 = LILayerNorm(d_model)
        self.norm2 = LILayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(LITransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def interpolated_forward(self, src, alphas, index, state1, state2, src_mask=None, src_key_padding_mask=None):
        ipjw_name = "transformer_encoder.layers.{}.self_attn.in_proj_weight".format(index)
        ipjb_name = "transformer_encoder.layers.{}.self_attn.in_proj_bias".format(index)
        out_prjw_name = "transformer_encoder.layers.{}.self_attn.out_proj.weight".format(index)
        out_prjb_name = "transformer_encoder.layers.{}.self_attn.out_proj.bias".format(index)

        src2 = self.self_attn.interpolated_forward(src, src, src, alphas, attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask,
                                                   ipjw1=state1[ipjw_name], ipjw2=state2[ipjw_name],
                                                   ipjb1=state1[ipjb_name], ipjb2=state2[ipjb_name],
                                                   out_prj1=state1[out_prjw_name], out_prj2=state2[out_prjw_name],
                                                   out_prjb1=state1[out_prjb_name], out_prjb2=state2[out_prjb_name],
                                                   qpjw1=None, qpjw2=None, kpjw1=None, kpjw2=None,
                                                   vpjw1=None, vpjw2=None, bias_k1=None, bias_k2=None,
                                                   bias_v1=None, bias_v2=None)[0]
        src = src + self.dropout1(src2)
        w_name = "transformer_encoder.layers.{}.norm1.weight".format(index)
        b_name = "transformer_encoder.layers.{}.norm1.bias".format(index)
        src = self.norm1.interpolated_forward(src, alphas, state1[w_name], state2[w_name],
                                              state1[b_name], state2[b_name])

        w_name = "transformer_encoder.layers.{}.linear1.weight".format(index)
        b_name = "transformer_encoder.layers.{}.linear1.bias".format(index)
        temp = self.linear1.interpolated_forward(src, alphas, state1[w_name], state2[w_name],
                                                 state1[b_name], state2[b_name])
        temp = self.dropout(self.activation(temp))

        w_name = "transformer_encoder.layers.{}.linear2.weight".format(index)
        b_name = "transformer_encoder.layers.{}.linear2.bias".format(index)
        src2 = self.linear2.interpolated_forward(temp, alphas, state1[w_name], state2[w_name],
                                                 state1[b_name], state2[b_name])

        src = src + self.dropout2(src2)
        w_name = "transformer_encoder.layers.{}.norm2.weight".format(index)
        b_name = "transformer_encoder.layers.{}.norm2.bias".format(index)
        src = self.norm2.interpolated_forward(src, alphas, state1[w_name], state2[w_name],
                                              state1[b_name], state2[b_name])
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LITransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(LITransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def interpolated_forward(self, src, alphas, state1, state2, mask=None, src_key_padding_mask=None):
        output = src

        i = 0
        for mod in self.layers:
            output = mod.interpolated_forward(output, alphas, i, state1, state2,
                                              src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            i += 1

        if self.norm is not None:
            output = self.norm(output)

        return output
