from ..layers.lstm import LILSTMCell
from ..layers.lstm import LILinear
from ..layers.embedding import LIEmbedding
from ..base import BaseModel

import torch.nn as nn


class LILSTM(BaseModel, nn.Module):
    def __init__(self, token_size, input_size, hidden_size, num_layers):
        super(LILSTM, self).__init__()
        self.n_layers = num_layers
        self.tie_weights = False

        self.encoder = LIEmbedding(token_size, input_size)
        self.decoder = LILinear(hidden_size, token_size)

        self.rnns = [
            LILSTMCell(input_size if l == 0 else hidden_size, hidden_size
                if l != self.n_layers - 1 else hidden_size)
                for l in range(self.n_layers)
        ]
        self.rnns = nn.ModuleList(self.rnns)
        self.n_inp = input_size
        self.n_hid = hidden_size
        self.n_token = token_size

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    @property
    def use_batchnorm(self):
        return False

    def reset_bn(self):
        pass

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.n_hid if l != self.n_layers - 1 else
        (self.n_inp if self.tie_weights else self.n_hid)).zero_(),
                 weight.new(1, bsz, self.n_hid if l != self.n_layers - 1 else
                 (self.n_inp if self.tie_weights else self.n_hid)).zero_())
                 for l in range(self.n_layers)]

    def forward(self, inputs, hx):
        emb = self.encoder(inputs)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hx[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            outputs.append(raw_output)

        hidden = new_hidden
        output = raw_output

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded, hidden

    def interpolated_forward(self, inputs, hx, alphas, state1, state2):
        w_name = "encoder.weight"
        emb = self.encoder.interpolated_forward(inputs, alphas,
                                                state1[w_name], state2[w_name])

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            i2h_w = "rnns.{}.i2h.weight".format(l)
            i2h_b = "rnns.{}.i2h.bias".format(l)
            h2h_w = "rnns.{}.h2h.weight".format(l)
            h2h_b = "rnns.{}.h2h.bias".format(l)

            raw_output, new_h = rnn.interpolated_forward(
                raw_output, hx[l], alphas,
                state1[i2h_w], state2[i2h_w],
                state1[i2h_b], state2[i2h_b],
                state1[h2h_w], state2[h2h_w],
                state1[h2h_b], state2[h2h_b],
            )

            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            outputs.append(raw_output)

        hidden = new_hidden
        output = raw_output

        w_name = "decoder.weight"
        b_name = "decoder.bias"
        decoded = self.decoder.interpolated_forward(output.view(output.size(0) * output.size(1), output.size(2)),
                                                    alphas,
                                                    state1[w_name], state2[w_name],
                                                    state1[b_name], state2[b_name])

        return decoded, hidden
