"""
losses.py — Perdas para o DeepfakeFusionModel.

Combina, no estilo do BreathNet:
    - CrossEntropy ponderada      : lida com o desbalanceamento bonafide vs spoof
    - CenterLoss (bonafide)       : compacta os embeddings bonafide num centro aprendido
    - ContrastLoss                : empurra os embeddings spoof para longe desse centro

Convenção de rótulo: y == 0 -> bonafide ; y >= 1 -> spoof (qualquer spoof_model).
O `embedding` vem de model(...)["embedding"].

Uso típico:
    crit = FeatureLoss(dim=cfg.d_model if cfg.fusion!='concat' else 2*cfg.d_model,
                       class_weights=torch.tensor([0.1, 0.9]))
    out = model(...)
    loss, parts = crit(out["logits"], out["embedding"], y)
    loss.backward()
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Aproxima embeddings da classe bonafide de um centro com atualização por momento.
    O centro NÃO é parâmetro otimizado pelo backprop; é uma média móvel (estável)."""

    def __init__(self, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _update_center(self, bona_emb: torch.Tensor):
        batch_mean = bona_emb.mean(dim=0)
        if not bool(self.initialized):
            self.center.copy_(batch_mean)
            self.initialized.fill_(True)
        else:
            self.center.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)

    def forward(self, emb: torch.Tensor, is_bonafide: torch.Tensor) -> torch.Tensor:
        bona = emb[is_bonafide]
        if bona.numel() == 0:
            return emb.sum() * 0.0  # mantém o grafo sem custo
        self._update_center(bona.detach())
        # 1 - cos_sim para puxar bonafide ao centro
        sim = F.cosine_similarity(bona, self.center.unsqueeze(0).expand_as(bona), dim=1)
        return (1.0 - sim).mean()


class ContrastLoss(nn.Module):
    """Empurra os embeddings spoof para longe do centro bonafide (separabilidade
    inter-classe). Usa o mesmo centro mantido pela CenterLoss."""

    def forward(self, emb: torch.Tensor, is_bonafide: torch.Tensor,
                center: torch.Tensor) -> torch.Tensor:
        fake = emb[~is_bonafide]
        if fake.numel() == 0:
            return emb.sum() * 0.0
        sim = F.cosine_similarity(fake, center.unsqueeze(0).expand_as(fake), dim=1)
        return (1.0 + sim).mean()  # minimiza similaridade (idealmente -1)


class FeatureLoss(nn.Module):
    """Perda total: CE ponderada + alpha*center + beta*contrast."""

    def __init__(
        self,
        dim: int,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.center = CenterLoss(dim, momentum)
        self.contrast = ContrastLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits: torch.Tensor, embedding: torch.Tensor, y: torch.Tensor):
        is_bona = (y == 0)
        l_ce = self.ce(logits, y)
        l_center = self.center(embedding, is_bona)
        l_contrast = self.contrast(embedding, is_bona, self.center.center)
        total = l_ce + self.alpha * l_center + self.beta * l_contrast
        parts = {"ce": l_ce.detach(), "center": l_center.detach(),
                 "contrast": l_contrast.detach(), "total": total.detach()}
        return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, C = 8, 256, 2
    logits = torch.randn(B, C, requires_grad=True)
    emb = torch.randn(B, D, requires_grad=True)
    y = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0])  # 3 bonafide, 5 spoof

    crit = FeatureLoss(dim=D, class_weights=torch.tensor([0.1, 0.9]), alpha=1.0, beta=1.0)
    loss, parts = crit(logits, emb, y)
    loss.backward()
    print({k: round(v.item(), 4) for k, v in parts.items()})
    print("grad logits ok:", logits.grad is not None, "| grad emb ok:", emb.grad is not None)
    print("OK")