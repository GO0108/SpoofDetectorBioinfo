"""
model.py — Detecção de deepfake de áudio por fusão de features.

Ramos:
    SSL (temporal)  : embeddings XLS-R/wav2vec2 -> SLS -> BiLSTM -> pooling
    handcrafted     : f0, jitter, shimmer, HNR -> atenção por feature -> MLP
    fusão           : "concat" (baseline) | "cross_attention" (BreathNet)
    cabeça          : MLP -> logits

Modos (ModelConfig.fusion): wav2vec_only | handcrafted_only | concat | cross_attention

Entradas do forward (passe só o que tiver):
    handcrafted        (B, F)            vetor acústico por utterance
    waveform           (B, N)            áudio cru 16 kHz  -> extração DENTRO do forward
                                         (normaliza + roda ssl_backbone aqui mesmo)
    waveform_mask      (B, N) bool       frames válidos do waveform (p/ padding)
    ssl_features       (B, T, D)         embeddings já extraídos (atalho/cache)
    ssl_hidden_states  (B, L, T, D)      todas as camadas -> SLS

Para o caminho `waveform`, injete um backbone HuggingFace:
    from embeddings import build_xlsr_backbone
    net = build_model(cfg, ssl_backbone=build_xlsr_backbone(freeze=cfg.freeze_ssl))
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    handcrafted_dim: int = 14          # nº de features acústicas (8 escalares + 6 agregados de F0)
    ssl_dim: int = 1024                # dim do XLS-R-300m (768 p/ base)
    d_model: int = 256
    fusion: str = "concat"             # wav2vec_only | handcrafted_only | concat | cross_attention
    use_sls: bool = True               # ponderação de camadas (quando há hidden_states/waveform)
    use_bilstm: bool = True
    bilstm_hidden: int = 128
    n_heads: int = 4
    num_classes: int = 2
    dropout: float = 0.2
    pool: str = "attentive"            # mean | attentive
    freeze_ssl: bool = True            # congela o backbone na extração em forward
    return_embedding: bool = True

    @property
    def needs_ssl(self) -> bool:
        return self.fusion in ("wav2vec_only", "concat", "cross_attention")

    @property
    def needs_handcrafted(self) -> bool:
        return self.fusion in ("handcrafted_only", "concat", "cross_attention")


# --------------------------------------------------------------------------- #
class SLS(nn.Module):
    """Sensitive Layer Selection: combina as camadas do transformer por pesos
    aprendidos (SLS/BreathNet). Devolve a sequência ponderada e os pesos."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, hidden_states: torch.Tensor):
        pooled = hidden_states.mean(dim=2)                        # (B, L, D)
        weights = torch.sigmoid(self.fc(pooled))                  # (B, L, 1)
        seq = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)  # (B, T, D)
        return seq, weights.squeeze(-1)


class AttentivePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = self.score(x).squeeze(-1)                        # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * attn).sum(dim=1)


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class TemporalBranch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.ssl_dim, cfg.d_model)
        if cfg.use_bilstm:
            self.bilstm = nn.LSTM(cfg.d_model, cfg.bilstm_hidden,
                                  batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * cfg.bilstm_hidden, cfg.d_model)
        else:
            self.bilstm, self.out = None, nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool = AttentivePool(cfg.d_model) if cfg.pool == "attentive" else None

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.dropout(F.relu(self.proj(seq)))
        if self.bilstm is not None:
            h, _ = self.bilstm(h)
            h = self.out(h)
        pooled = self.pool(h, mask) if self.pool is not None else _masked_mean(h, mask)
        return pooled, h


class HandcraftedBranch(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.feat_attn = nn.Linear(cfg.handcrafted_dim, cfg.handcrafted_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.handcrafted_dim, cfg.d_model), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor):
        weights = torch.softmax(self.feat_attn(x), dim=-1)        # (B, F) importância
        return self.mlp(x * weights), weights


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, hand_vec, ssl_seq, mask: Optional[torch.Tensor] = None):
        q = hand_vec.unsqueeze(1)
        key_padding = (~mask) if mask is not None else None
        out, attn_w = self.attn(q, ssl_seq, ssl_seq, key_padding_mask=key_padding)
        return self.norm(out.squeeze(1) + hand_vec), attn_w.squeeze(1)


# --------------------------------------------------------------------------- #
class DeepfakeFusionModel(nn.Module):
    def __init__(self, cfg: ModelConfig, ssl_backbone: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.ssl_backbone = ssl_backbone           # HF Wav2Vec2Model p/ extração em forward
        if cfg.needs_ssl:
            self.sls = SLS(cfg.ssl_dim) if cfg.use_sls else None
            self.temporal = TemporalBranch(cfg)
        if cfg.needs_handcrafted:
            self.handcrafted = HandcraftedBranch(cfg)
        if cfg.fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(cfg)
        head_in = cfg.d_model * (2 if cfg.fusion == "concat" else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.d_model), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_classes),
        )

    # ---- extração XLS-R DENTRO do forward ---------------------------------- #
    def _run_backbone(self, waveform, waveform_mask):
        if self.ssl_backbone is None:
            raise ValueError("waveform fornecido mas ssl_backbone é None "
                             "(use build_xlsr_backbone e passe a build_model).")
        x = waveform
        if waveform_mask is not None:                         # normalização por amostra (mask-aware)
            m = waveform_mask.float()
            n = m.sum(1, keepdim=True).clamp_min(1.0)
            mean = (x * m).sum(1, keepdim=True) / n
            var = (((x - mean) ** 2) * m).sum(1, keepdim=True) / n
            x = ((x - mean) / torch.sqrt(var + 1e-7)) * m
            attn = waveform_mask.long()
        else:
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            attn = None
        ctx = torch.no_grad() if self.cfg.freeze_ssl else contextlib.nullcontext()
        with ctx:
            out = self.ssl_backbone(x, attention_mask=attn, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)            # (B, L, T, D)

        frame_mask = None
        if attn is not None and hasattr(self.ssl_backbone, "_get_feat_extract_output_lengths"):
            in_len = waveform_mask.long().sum(1)
            out_len = self.ssl_backbone._get_feat_extract_output_lengths(in_len).long()
            T = hs.shape[2]
            frame_mask = torch.arange(T, device=hs.device)[None, :] < out_len[:, None]
        return hs, frame_mask

    def _resolve_ssl_sequence(self, waveform, waveform_mask, ssl_features, ssl_hidden_states):
        layer_weights = frame_mask = None
        if waveform is not None:
            hs, frame_mask = self._run_backbone(waveform, waveform_mask)
            if self.sls is not None:
                seq, layer_weights = self.sls(hs)
            else:
                seq = hs[:, -1]
        elif ssl_hidden_states is not None:
            if self.sls is not None:
                seq, layer_weights = self.sls(ssl_hidden_states)
            else:
                seq = ssl_hidden_states[:, -1]
        elif ssl_features is not None:
            seq = ssl_features
        else:
            raise ValueError("Modo SSL exige waveform, ssl_features ou ssl_hidden_states.")
        return seq, layer_weights, frame_mask

    def forward(self, handcrafted=None, ssl_features=None, ssl_hidden_states=None,
                waveform=None, waveform_mask=None, ssl_mask=None) -> dict:
        cfg, info = self.cfg, {}
        ssl_pooled = ssl_seq = hand_vec = None

        if cfg.needs_ssl:
            seq, layer_w, frame_mask = self._resolve_ssl_sequence(
                waveform, waveform_mask, ssl_features, ssl_hidden_states)
            if ssl_mask is None:
                ssl_mask = frame_mask                          # máscara derivada do backbone
            ssl_pooled, ssl_seq = self.temporal(seq, ssl_mask)
            if layer_w is not None:
                info["layer_weights"] = layer_w
        if cfg.needs_handcrafted:
            if handcrafted is None:
                raise ValueError("fusion requer 'handcrafted' mas recebeu None.")
            hand_vec, info["feature_weights"] = self.handcrafted(handcrafted)

        if cfg.fusion == "wav2vec_only":
            embedding = ssl_pooled
        elif cfg.fusion == "handcrafted_only":
            embedding = hand_vec
        elif cfg.fusion == "concat":
            embedding = torch.cat([ssl_pooled, hand_vec], dim=-1)
        elif cfg.fusion == "cross_attention":
            embedding, info["cross_attn"] = self.fusion(hand_vec, ssl_seq, ssl_mask)
        else:
            raise ValueError(f"fusion desconhecida: {cfg.fusion}")

        out = {"logits": self.head(embedding)}
        if cfg.return_embedding:
            out["embedding"] = embedding
        out.update(info)
        return out


def build_model(cfg: Optional[ModelConfig] = None, ssl_backbone=None) -> DeepfakeFusionModel:
    return DeepfakeFusionModel(cfg or ModelConfig(), ssl_backbone)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, F_DIM, D_SSL, L = 4, 50, 14, 1024, 25
    handcrafted = torch.randn(B, F_DIM)
    ssl_features = torch.randn(B, T, D_SSL)
    ssl_hidden = torch.randn(B, L, T, D_SSL)
    mask = torch.ones(B, T, dtype=torch.bool); mask[0, 30:] = False

    for fusion in ("wav2vec_only", "handcrafted_only", "concat", "cross_attention"):
        cfg = ModelConfig(handcrafted_dim=F_DIM, ssl_dim=D_SSL, fusion=fusion)
        out = build_model(cfg)(handcrafted=handcrafted, ssl_features=ssl_features, ssl_mask=mask)
        print(f"[{fusion:16s}] logits={tuple(out['logits'].shape)}")

    # --- caminho waveform: extração DENTRO do forward (backbone de mentira) --- #
    class FakeBackbone(nn.Module):
        STRIDE = 320
        def __init__(self, dim, n_layers=25):
            super().__init__()
            self.dim, self.n_layers = dim, n_layers
            self.lin = nn.Linear(1, dim)
        def _get_feat_extract_output_lengths(self, lengths):
            return torch.div(lengths, self.STRIDE, rounding_mode="floor")
        def forward(self, input_values, attention_mask=None, output_hidden_states=False):
            B, N = input_values.shape
            Tt = N // self.STRIDE
            base = self.lin(input_values[:, :Tt * self.STRIDE:self.STRIDE].unsqueeze(-1))
            hs = tuple(base + i for i in range(self.n_layers))
            return type("O", (), {"hidden_states": hs, "last_hidden_state": hs[-1]})()

    N = 16000
    wav = torch.randn(B, N)
    wmask = torch.ones(B, N, dtype=torch.bool); wmask[0, N // 2:] = False
    cfg = ModelConfig(handcrafted_dim=F_DIM, ssl_dim=D_SSL, fusion="concat", use_sls=True)
    net = build_model(cfg, ssl_backbone=FakeBackbone(D_SSL))
    out = net(handcrafted=handcrafted, waveform=wav, waveform_mask=wmask)
    print(f"[waveform->forward] logits={tuple(out['logits'].shape)}  "
          f"layer_weights={tuple(out['layer_weights'].shape)}")
    print("OK")