import torch
from torch import nn
import torch.nn.functional as F
from kirchhoff_gn import WeightedGNN
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
        # self.rnn = rnntype(
        #    n_embed, n_hidden, n_layers, bidirectional=bidirectional, dropout=dropout
        # )
        self.transform = nn.Linear(n_hidden, n_layers * n_hidden)

        self.gn = WeightedGNN(
            input_size=n_embed,
            hidden_size=n_hidden,
            ntokens=len(self.vocab),
            padding_idx=self.vocab.pad(),
            snip_start_end=False
        )

    def forward(self, data, lens=None):
        final_state, \
        (flattened_internal,
         flattened_internal,
         flattened_internal_mask,
         external_key, external_val, external_idxs, mask) = self.gn(data)
        return flattened_internal, final_state
        """
        if len(data.shape) == 3:
            emb    = torch.matmul(data, self.embed.weight)
            tokens = torch.argmax(data.detach(),dim=-1)
            emb    = emb * (tokens != self.vocab.pad()).unsqueeze(2).float()
        else:
            # Will take this path usually.
            emb   = self.embed(data)
        if lens is not None:
            padded_sequence = self.embed_dropout(emb)
            total_length = padded_sequence.shape[0]
            packed_sequence = nn.utils.rnn.pack_padded_sequence(padded_sequence,
                                                                lens.cpu())
            packed_output, hidden = self.rnn(packed_sequence)
            output_padded,_ = nn.utils.rnn.pad_packed_sequence(packed_output,
                                                               total_length=total_length,
                                                               padding_value=self.vocab.pad())
            return output_padded, hidden
        else:
            return self.rnn(self.embed_dropout(emb))
        """
