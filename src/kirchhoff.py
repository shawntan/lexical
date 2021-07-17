import torch
from torch import nn as nn
from torch.nn import functional as F

DEBUG = False

EPS = torch.finfo(torch.float32).tiny

class KirchhoffNormalisation(nn.Module):
    def __init__(self, dropout, plus_one=False, smoothing_eps=5e-4,
                 tril=False):
        super(KirchhoffNormalisation, self).__init__()
        self.plus_one = plus_one
        self.dropout = dropout
        self.neg_inf = torch.log(torch.tensor(EPS))
        self.smoothing_eps = smoothing_eps
        self.max_trick = False
        self.no_root_score = True

        self.tril = tril

    def forward(self, head_dep_score, head_root_score, mask,
                entropy=True):
        length_mask = ~mask[:, :, None] | ~mask[:, None, :]
        eye = torch.eye(head_root_score.size(1),
                        device=length_mask.device)
        diagonal_mask = eye.bool()
        idxs = torch.arange(head_root_score.size(1),
                            dtype=torch.long,
                            device=length_mask.device)
        dep_mask = diagonal_mask | length_mask

        if self.plus_one:
            head_dep_score = F.pad(head_dep_score, (0, 1), 'constant', 0.)
            root_dep_score = F.pad(head_root_score, (0, 1), 'constant', 0.)

        head_dep_score = head_dep_score.masked_fill(dep_mask[..., None], self.neg_inf)
        head_root_score = head_root_score.masked_fill(~mask[..., None], self.neg_inf)

        if self.training:
            pred_dep_mask = torch.rand_like(head_dep_score)  < self.dropout
            pred_root_mask = torch.rand_like(head_root_score) < self.dropout
            head_dep_score = head_dep_score.masked_fill(pred_dep_mask, self.neg_inf)
            head_root_score = head_root_score.masked_fill(pred_root_mask, self.neg_inf)

        if self.max_trick:
            k = torch.maximum(
                torch.amax(
                    torch.logsumexp(head_dep_score, dim=(-2, -1)),
                    dim=-1
                ),
                torch.amax(head_root_score, dim=(-2, -1))
            )
            head_dep_score = head_dep_score - k[:, None, None, None]
            head_root_score = head_root_score - k[:, None, None]

        with torch.enable_grad():
            if self.no_root_score:
                head_root_score = 0 * head_root_score
            if not self.training and not head_dep_score.requires_grad:
                head_dep_score.requires_grad = True
                head_root_score.requires_grad = True
            dep_score = torch.logsumexp(head_dep_score, dim=-1)
            root_score = torch.logsumexp(head_root_score, dim=-1)

            A = torch.exp(dep_score)
            A = A + self.smoothing_eps
            A = A.masked_fill(dep_mask, 0.)
            if self.training:
                A = A / (1 - self.dropout)
            if self.no_root_score:
                rho = torch.exp(root_score)
                rho = rho.masked_fill(~mask, 0.)
            else:
                rho = torch.exp(root_score)
                rho = rho + self.smoothing_eps
                rho = rho.masked_fill(~mask, 0.)
                if self.training:
                    rho = rho / (1 - self.dropout)
            if self.tril:
                A = torch.tril(A)

            L = torch.diag_embed(torch.sum(A, dim=-1)) - A
            L[:, :, 0] = rho
            L = L.masked_fill(length_mask, 0.) \
                 .masked_fill((diagonal_mask[None, :] &
                               ~mask[:, None, :]), 1.) #  'eye' the padding
            logdet_L = torch.logdet(L)
            head_parents, head_root = torch.autograd.grad(
                torch.sum(logdet_L),
                (head_dep_score, head_root_score),
                create_graph=True
            )
            head_parents = head_parents.masked_fill(dep_mask[..., None], 0.)
            parents_ = head_parents.sum(-1)

        if DEBUG:
            if (parents_ < 0.).any():
                violating = (parents_.detach() < 0.)
                print("parents < 0")
                print(parents_[violating])
                print(A[violating])
                print(dep_score[violating])
                # head_parents[violating] = 0.

            if (parents_ > 1.).any():
                violating = (parents_.detach() > 1.)
                print("parents > 1")
                print(parents_[violating])
                print(A[violating])

            if (~torch.isfinite(parents_)).any():
                violating = (~torch.isfinite(parents_.detach()))
                print("parents nan")
                print(parents_[violating])
                print(A[violating])

        if self.training:
            head_parents = head_parents.masked_fill(pred_dep_mask, 0.)
        if self.plus_one:
            head_parents = head_parents[..., :-1]
        if entropy:
            entropy = logdet_L - torch.einsum('bij,bij->b', parents_, dep_score)
            return head_parents, head_root, entropy
        else:
            return head_parents, head_root
