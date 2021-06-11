import torch.nn.functional as F

from torch.nn.modules.activation import *


class LIMultiheadAttention(MultiheadAttention):
    def interpolated_forward(self, query, key, value, alpha, ipjw1, ipjw2, qpjw1, qpjw2, kpjw1, kpjw2,
                             vpjw1, vpjw2, ipjb1, ipjb2, bias_k1, bias_k2, bias_v1, bias_v2,
                             out_prj1, out_prj2, out_prjb1, out_prjb2,
                             key_padding_mask=None, need_weights=True, attn_mask=None):
        if ipjw1 is not None:
            ipjw = (1 - alpha) * ipjw1 + alpha * ipjw2
        else:
            ipjw = None

        if qpjw1 is not None:
            qpjw = (1 - alpha) * qpjw1 + alpha * qpjw2
        else:
            qpjw = None

        if kpjw1 is not None:
            kpjw = (1 - alpha) * kpjw1 + alpha * kpjw2
        else:
            kpjw = None

        if vpjw1 is not None:
            vpjw = (1 - alpha) * vpjw1 + alpha * vpjw2
        else:
            vpjw = None

        if ipjb1 is not None:
            ipjb = (1 - alpha) * ipjb1 + alpha * ipjb2
        else:
            ipjb = None

        if bias_k1 is not None:
            bias_k = (1 - alpha) * bias_k1 + alpha * bias_k2
        else:
            bias_k = None

        if bias_v1 is not None:
            bias_v = (1 - alpha) * bias_v1 + alpha * bias_v2
        else:
            bias_v = None

        if out_prj1 is not None:
            out_prj = (1 - alpha) * out_prj1 + alpha * out_prj2
        else:
            out_prj = None

        if out_prjb1 is not None:
            out_projb = (1 - alpha) * out_prjb1 + alpha * out_prjb2
        else:
            out_projb = None

        if not self._qkv_same_embed_dim:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                ipjw, ipjb,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_prj, out_projb,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=qpjw, k_proj_weight=kpjw,
                v_proj_weight=vpjw)
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                ipjw, ipjb,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_prj, out_projb,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
