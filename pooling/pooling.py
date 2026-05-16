import re
import torch
import torch.nn as nn
import torch
import numpy as np
import re



class Pooling(nn.Module):
    def __init__(self,
                strategy='mean',
                padding_side = 'right'):
        super(Pooling, self).__init__()
        self.strategy = strategy
        self.padding_side = padding_side

    
    def forward(self, 
                token_embeddings,
                embed_mask,
                weight = None,
                ):
        embed_mask = embed_mask.to(token_embeddings.device)        
        if self.strategy == 'mean':
            pooled = torch.sum(token_embeddings * embed_mask.unsqueeze(-1), dim=1) / torch.sum(embed_mask, dim=1).unsqueeze(-1)
            pooled.masked_fill_(torch.isnan(pooled), 0)
        elif self.strategy == 'last':
            def _extract_last_nonzero(m):
                nonzeros = (m == 1).nonzero(as_tuple=True)[0]
                return torch.max(nonzeros) if nonzeros.size(0) > 0 else 0

            if self.padding_side == 'right':
                last_indices = torch.tensor([_extract_last_nonzero(m) for m in embed_mask])
            elif self.padding_side == 'left':
                last_indices = torch.full((embed_mask.size(0),), -1, dtype=torch.long, device=embed_mask.device)
            else:
                raise ValueError(f"Unknown padding side: {self.padding_side}")

            i = torch.arange(token_embeddings.shape[0]).reshape(token_embeddings.shape[0], 1, 1)
            j = last_indices.reshape(last_indices.shape[0], 1, 1)
            k = torch.arange(token_embeddings.shape[2])
            pooled = token_embeddings[i, j, k][:, 0, :]
            pooled.masked_fill_(torch.isnan(pooled), 0)
        elif self.strategy == 'weighted_mean':
            if weight is None:
                raise ValueError("Weight tensor must be provided for 'weighted_mean' strategy")
            weight = weight.to(token_embeddings.device)
            weighted_embeddings = token_embeddings * weight.unsqueeze(-1)
            pooled = torch.sum(weighted_embeddings * embed_mask.unsqueeze(-1), dim=1) / torch.sum(weight * embed_mask, dim=1).unsqueeze(-1)
            pooled.masked_fill_(torch.isnan(pooled), 0)
        else:
            raise ValueError(f'Unknown pooling strategy: {self.strategy}')
        return pooled

