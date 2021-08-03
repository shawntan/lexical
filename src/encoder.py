import torch
from torch import nn
import torch.nn.functional as F
from .kirchhoff_gn import WeightedGNN

class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
            n_embed,
            n_hidden,
            n_layers,
            bidirectional=True,
            dropout=0,
            rnntype=nn.LSTM,
    ):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.embed_dropout = nn.Dropout(dropout)
        self.transform = nn.Linear(n_hidden, n_layers * n_hidden)
        self.n_layers = n_layers

        self.gn = WeightedGNN(
            input_size=n_embed,
            hidden_size=n_hidden,
            ntokens=len(self.vocab),
            padding_idx=self.vocab.pad(),
            parser_seq_processor='rnn',
            snip_start_end=True,
            parser_dropout=0.1,
            dropout=0.1
        )

    def forward(self, data, lens=None):
        final_state, \
        (flattened_internal,
         flattened_internal,
         flattened_internal_mask,
         external_key, external_val, external_idxs, mask) = self.gn(data)
        flattened_internal = F.pad(flattened_internal, (0, 0, 0, 0, 1, 1))
        final_state = self.transform(final_state).view(final_state.size(0),
                                                       self.n_layers,
                                                       final_state.size(1))

        final_state = torch.tan(final_state).permute(1, 0, 2)
        final_state = (final_state, torch.zeros_like(final_state))
        return (flattened_internal, final_state)
