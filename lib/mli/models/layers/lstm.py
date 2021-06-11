from .linear import LILinear
import torch.nn as nn
import torch


class LILSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LILSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.i2h = LILinear(self.input_size, 4 * self.hidden_size)
        self.h2h = LILinear(self.input_size, 4 * self.hidden_size)

    def forward(self, inputs, hx):
        hidden_list = []
        n_hid = self.hidden_size
        h, cell = hx

        h = h.squeeze(0)
        cell = cell.squeeze(0)

        for i in range(len(inputs)):
            x = inputs[i]

            x_components = self.i2h(x)
            h_components = self.h2h(h)

            preactivations = x_components + h_components

            gates_together = torch.sigmoid(preactivations[:, 0:3 * n_hid])
            forget_gate = gates_together[:, 0:n_hid]
            input_gate = gates_together[:, n_hid:2 * n_hid]
            output_gate = gates_together[:, 2 * n_hid:3 * n_hid]
            new_cell = torch.tanh(preactivations[:, 3 * n_hid:4 * n_hid])

            cell = forget_gate * cell + input_gate * new_cell
            h = output_gate * torch.tanh(cell)
            hidden_list.append(h)
        hidden_stacked = torch.stack(hidden_list)
        return hidden_stacked, (h.unsqueeze(0), cell.unsqueeze(0))

    def interpolated_forward(self, inputs, hx, alpha, i2h_w1, i2h_w2, i2h_b1, i2h_b2,
                             h2h_w1, h2h_w2, h2h_b1, h2h_b2):
        hidden_list = []
        n_hid = self.hidden_size
        h, cell = hx

        h = h.squeeze(0)
        cell = cell.squeeze(0)

        for i in range(len(inputs)):
            x = inputs[i]

            x_components = self.i2h.interpolated_forward(x,
                                                         alpha,
                                                         i2h_w1,
                                                         i2h_w2,
                                                         i2h_b1,
                                                         i2h_b2)
            h_components = self.h2h.interpolated_forward(h,
                                                         alpha,
                                                         h2h_w1,
                                                         h2h_w2,
                                                         h2h_b1,
                                                         h2h_b2)
            preactivations = x_components + h_components

            gates_together = torch.sigmoid(preactivations[:, 0:3 * n_hid])
            forget_gate = gates_together[:, 0:n_hid]
            input_gate = gates_together[:, n_hid:2 * n_hid]
            output_gate = gates_together[:, 2 * n_hid:3 * n_hid]
            new_cell = torch.tanh(preactivations[:, 3 * n_hid:4 * n_hid])

            cell = forget_gate * cell + input_gate * new_cell
            h = output_gate * torch.tanh(cell)
            hidden_list.append(h)
        hidden_stacked = torch.stack(hidden_list)
        return hidden_stacked, (h.unsqueeze(0), cell.unsqueeze(0))
