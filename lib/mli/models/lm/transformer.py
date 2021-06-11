import math
import torch

import torch.nn as nn

from mli.models.layers import LILinear, LIEmbedding
from mli.models import BaseModel
from mli.models.layers import LITransformerEncoder, LITransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LITransformerModel(BaseModel, nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(LITransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError("TransformerEncoder module does not exist in PyTorch 1.1 or lower.")
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = LITransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = LITransformerEncoder(encoder_layers, nlayers)
        self.encoder = LIEmbedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = LILinear(ninp, ntoken)

        self.init_weights()

    @property
    def use_batchnorm(self):
        return False

    def reset_bn(self):
        pass

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def interpolated_forward(self, src, alphas, state1, state2, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        w_name = "encoder.weight"
        src = self.encoder.interpolated_forward(src, alphas, state1[w_name], state2[w_name]) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder.interpolated_forward(src, alphas,
                                                               state1, state2,
                                                               self.src_mask)

        w_name = "decoder.weight"
        b_name = "decoder.bias"

        output = self.decoder.interpolated_forward(output, alphas,
                                                   state1[w_name], state2[w_name],
                                                   state1[b_name], state2[b_name])
        return output
