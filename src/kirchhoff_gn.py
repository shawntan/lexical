import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .kirchhoff import KirchhoffNormalisation
from . import hinton


class Conv1d(nn.Module):
    """1D convolution layer."""

    def __init__(self, hidden_size, kernel_size, dilation=1):
        """Initialization.

        Args:
          hidden_size: dimension of input embeddings
          kernel_size: convolution kernel size
          dilation: the spacing between the kernel points
        """
        super(Conv1d, self).__init__()

        if kernel_size % 2 == 0:
            padding = (kernel_size // 2) * dilation
            self.shift = True
        else:
            padding = ((kernel_size - 1) // 2) * dilation
            self.shift = False
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            padding=padding,
            dilation=dilation)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x):
        """Compute convolution.

        Args:
          x: input embeddings
        Returns:
          conv_output: convolution results
        """

        if self.shift:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:]
        else:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)


class WeightedGNNLayer(nn.Module):
    def __init__(self,
                 hidden_dim, nrels,
                 dropout=0.1, dropatt=0.1):
        super(WeightedGNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nrels = nrels
        self.proj_nbrs = nn.Linear(hidden_dim, nrels * hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def shrink(self, x):
        return x - torch.tanh(x)

    def forward(self, x, h_prev, attn_mask):
        batch_size, length, hdim = x.size()
        h_prev = self.dropout(h_prev)
        proj_h_prev = self.proj_nbrs(h_prev)
        proj_h_prev = F.layer_norm(
            proj_h_prev.view(batch_size, length, self.nrels, hdim),
            normalized_shape=(hdim,)
        )
        lin_sum = torch.einsum('bijh,bihd->bjd', attn_mask, proj_h_prev)
        shift, scale, proj_x = self.in_proj(x).chunk(3, dim=-1)
        h = F.relu(
            shift * F.layer_norm(
                proj_x + self.shrink(lin_sum),
                normalized_shape=(hdim,),
            ) + scale
        )
        y = torch.tanh(self.out_proj(h))
        # g = torch.sigmoid(lin_g)
        # y = g * torch.tanh(lin_y) + (1 - g) * torch.tanh(x)
        return y


class WeightedGNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 ntokens, padding_idx,
                 snip_start_end=False,
                 dropout=0.2, dropoutm=0.1,
                 parser_dropout=0.05,
                 parser_seq_processor='rnn',
                 prefix_attachment=False,
                 gumbel_noise=True):
        super(WeightedGNN, self).__init__()

        self.snip_start_end = snip_start_end
        self.padding_idx = padding_idx

        self.emb_syn = nn.Embedding(ntokens, input_size, padding_idx=self.padding_idx)
        self.emb_sem = nn.Embedding(ntokens, input_size, padding_idx=self.padding_idx)
        self.emb_aux = nn.Embedding(ntokens, input_size, padding_idx=self.padding_idx)

        self.parse = NRIParser(input_size, num_layers=2,
                            dropout=parser_dropout,
                            parser_seq_processor=parser_seq_processor,
                            prefix_attachment=prefix_attachment,
                            gumbel_noise=gumbel_noise)

        self.compute_layer = WeightedGNNLayer(hidden_dim=input_size, nrels=8,
                                              dropout=dropout, dropatt=0.1)
        # self.compute_layer = GraphBertLayer(hidden_dim=input_size, nrels=8,
        #                                     dropout=dropout, dropatt=0.1)
    def parse_only(self, input):
        A_rels, X_aux, X_sem, lengths, mask, parser_h, root_rels = \
            self.initial_processing(input)
        return A_rels



    def forward(self, input):
        A_rels, X_aux, X_sem, lengths, mask, \
        parser_h, root_rels, external_idxs = self.initial_processing(input)
        root = root_rels.sum(-1)
        prev_level = torch.zeros_like(X_sem)
        for i in range(root.size(1), 0, -1):
            # print((i > lengths).long())
            prev_level = self.compute_layer(X_sem, prev_level, attn_mask=A_rels)
            prev_level = prev_level.masked_fill((i > lengths)[:, None, None], 0.)
        # print(prev_level)
        # print(root)
        final_state = torch.einsum('bih,bi->bh', prev_level, root)
        flattened_internal = prev_level.transpose(1, 0)
        flattened_internal_mask = mask
        external_key = parser_h.transpose(1, 0)
        external_val = X_aux.transpose(1, 0)

        return final_state,\
            (flattened_internal,
             flattened_internal,
             flattened_internal_mask,
             external_key, external_val, external_idxs, mask)

    def initial_processing(self, input):
        max_length, batch_size = input.size()
        if self.snip_start_end:
            input = input.clone()
            input = input[1:]
            mask = (input != self.padding_idx)[1:]
            lengths = mask.sum(0)
            input[lengths, torch.arange(batch_size)] = self.padding_idx
            input = input[:-1]
        else:
            mask = input != self.padding_idx
            lengths = mask.sum(0)
        mask_ = mask.t()
        # X = self.embedding(input)
        # X_ = X.transpose(1, 0)
        # X_aux = X_.chunk(3, dim=-1)
        input_ = input.t()
        X_sem = self.emb_sem(input_)
        X_syn = self.emb_syn(input_)
        X_aux = self.emb_aux(input_)
        A_rels, root_rels, parser_h = self.parse(X_syn, mask_)
        return A_rels, X_aux, X_sem, lengths, mask, parser_h, root_rels, input


class RelativeEncodingScore(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1, max_len=500):
        super(RelativeEncodingScore, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len * 2 + 1, input_size)
        position = -(torch.arange(max_len * 2 + 1) - max_len)
        div_term = torch.exp(torch.arange(0, input_size, 2, dtype=torch.float) *
                             (-math.log(10000.0) / input_size))

        vals = position[:, None] * div_term[None, :]
        pe[:, 0::2] = torch.sin(vals)
        pe[:, 1::2] = torch.cos(vals)
        # pe = position[:, None].repeat(1, d_model)
        self.register_buffer('pe', pe)
        self.max_len = max_len
        self.input_size = input_size
        self.output_size = output_size
        self.transform = nn.Linear(input_size, output_size)

    def forward(self, l):
        pos = self.transform(self.pe)
        relpos = pos.as_strided(
            (l, l, self.output_size),
            (self.output_size, self.output_size, 1),
            (self.max_len - l + 1) * self.output_size
        )[:, torch.arange(l - 1, -1, -1)]
        return relpos


class Parser(nn.Module):
    def __init__(self, input_size, num_layers=3, conv_size=3,
                 dropout=0.2,
                 parser_seq_processor='rnn',
                 prefix_attachment=True,
                 gumbel_noise=True):
        super(Parser, self).__init__()
        self.num_layers = num_layers

        self.bilinear_dim = 10
        self.rel_types = 8
        self.type = parser_seq_processor
        self.gumbel_noise = gumbel_noise

        self.parser_ff = nn.Sequential(
            nn.Linear(input_size, self.bilinear_dim),
        )

        self.relpos = RelativeEncodingScore(input_size, input_size // 4)
        self.parser_drop = nn.Dropout(dropout)
        self.parser_c = nn.Linear(input_size, input_size // 4)
        self.parser_p = nn.Linear(input_size, input_size // 4)
        self.parser_cumul = nn.Linear(input_size, input_size // 4)
        self.parser_act = nn.Sequential(
            nn.LayerNorm(input_size // 4),
            nn.Tanh(),
            self.parser_drop
        )
        self.parser_o = nn.Linear(input_size // 4, self.rel_types)
        nn.init.zeros_(self.parser_o.weight)
        nn.init.zeros_(self.parser_o.bias)

        self.relaxed_structure = KirchhoffNormalisation(
            dropout=0.01,
            smoothing_eps=0.,
            plus_one=False, tril=prefix_attachment
        )
        if self.type == 'conv':
            self.parser_layers = nn.ModuleList([
                nn.Sequential(
                    Conv1d(input_size, conv_size),
                    nn.LayerNorm(input_size, elementwise_affine=False),
                    nn.Tanh()
                ) for i in range(num_layers)])
        elif self.type == 'rnn':
            self.parser_layers = nn.LSTM(input_size, input_size // 2, num_layers,
                                         dropout=dropout, bidirectional=True)

    def forward(self, X, mask):
        lengths = mask.sum(1)
        visibility = mask[:, :, None] & mask[:, None, :]
        h = X

        cumul_h = F.pad(
            torch.cumsum(
                self.parser_cumul(h) \
                    .masked_fill(mask[..., None], 0.),
                dim=1
            ), (0, 0, 1, 0)
        )
        f_cumul_h = cumul_h[:, :-1]
        b_cumul_h = cumul_h[:, -1, None] - cumul_h[:, 1:]
        cumul_lin = (f_cumul_h[:, :, None] - b_cumul_h[:, None, :])

        if self.type == 'conv':
            for i in range(self.num_layers):
                h_in = h.masked_fill(~mask[:, :, None], 0)
                h = self.parser_layers[i](h_in)
                h = self.parser_drop(h)

        elif self.type == 'rnn':
            h = pack_padded_sequence(h, lengths.cpu(), enforce_sorted=False,
                                     batch_first=True)
            h, _ = self.parser_layers(h)
            h, _ = pad_packed_sequence(h, batch_first=True)
            h = self.parser_drop(h)

        c_lin = self.parser_c(h)
        p_lin = self.parser_p(h)


        hid = self.parser_act(
            c_lin[:, :, None] +
            p_lin[:, None, :] +
            cumul_lin +
            self.relpos(h.size(1))[None]
        )
        logits_rels = self.parser_o(hid)
        logits_rels = torch.tanh(logits_rels / 7.5) * 7.5

        if self.training and self.gumbel_noise:
            logits_rels = (logits_rels +
                -torch.log(
                    -torch.log(
                        torch.rand_like(logits_rels).clamp(min=1e-6)))
            )
        logits_rels = logits_rels.masked_fill(~visibility[..., None], -64)
        root_logits_rels = torch.diagonal(logits_rels, dim1=1, dim2=2).permute(0, 2, 1)
        p, p_root, entrpy = self.relaxed_structure.forward(
            head_dep_score=logits_rels,
            head_root_score=0 * root_logits_rels,
            mask=mask
        )
        return p, p_root, h

class NRILayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, transform_edge=True):
        super(NRILayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_e = nn.Linear(hidden_dim, hidden_dim)
        self.out_e = nn.Linear(hidden_dim, hidden_dim)
        self.activation_e = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        if transform_edge:
            self.e2e = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.e2e = None

        self.e2v = nn.Linear(hidden_dim, hidden_dim)
        self.activation_v = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, v, e=None, v_mask=None, e_mask=None):
        diag_mask = torch.eye(v.size(1), dtype=torch.bool, device=v.device)
        lin_e = self.in_e(v)[:, :, None] + self.out_e(v)[:, None, :]
        if e is not None:
            lin_e += self.e2e(e)
        e = self.activation_e(lin_e) \
            .masked_fill(diag_mask[None, ..., None], 0.)

        if v_mask is not None:
            v = v.masked_fill(v_mask[..., None], 0.)
        if e_mask is not None:
            e = e.masked_fill(e_mask[..., None], 0.)

        v = v + self.activation_v(torch.sum(e, dim=1))
        return v, e

class NRIParser(nn.Module):
    def __init__(self, input_size, num_layers=3,
                 prefix_attachment=True,
                 conv_size=3,
                 dropout=0.2,
                 parser_seq_processor='rnn',
                 gumbel_noise=True):
        super(NRIParser, self).__init__()
        self.num_layers = num_layers
        self.rel_types = 8

        self.layers = nn.ModuleList([
            NRILayer(hidden_dim=input_size)
            for l in range(num_layers)
        ])
        self.relaxed_structure = KirchhoffNormalisation(
            dropout=0.01,
            smoothing_eps=0.,
            plus_one=False, tril=prefix_attachment
        )
        self.parser_o = nn.Linear(input_size, self.rel_types)

    def forward(self, X, mask):
        lengths = mask.sum(1)
        e_mask = mask[:, :, None] & mask[:, None, :]
        v, e = X, None
        for layer in self.layers:
            v, e = layer.forward(v, e, v_mask=mask, e_mask=e_mask)

        logits_rels = self.parser_o(e)
        root_logits_rels = torch.diagonal(logits_rels, dim1=1, dim2=2).permute(0, 2, 1)
        p, p_root, entrpy = self.relaxed_structure.forward(
            head_dep_score=logits_rels,
            head_root_score=0 * root_logits_rels,
            mask=mask
        )
        # print("v", v.size())
        # print("e", e.size())
        return p, p_root, v


if __name__ == "__main__":
    length = 9
    x = torch.randint(low=1, high=20, size=(length, 5))
    x[-8:, 0] = 0
    x[-2:, 1] = 0
    x[-4:, 2] = 0
    mask = x != 0
    print(mask.long())
    gn = WeightedGNN(
        input_size=80, hidden_size=80,
        ntokens=20, padding_idx=0,
        snip_start_end=False
    )
    gn(x)
