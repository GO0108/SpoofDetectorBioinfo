"""
train.py — Treino e avaliação do DeepfakeFusionModel no BRSpeech-DF.

Dois modos:
    (padrão)  treino com split agrupado por locutor + validação por EER/AUC
    --lomo    leave-one-spoof_model-out: tabela de generalização p/ gerador não visto

Exemplos:
    python train.py /caminho/dos/chunks --fusion concat --epochs 30
    python train.py /caminho/dos/chunks --fusion handcrafted_only --feature-loss
    python train.py /caminho/dos/chunks --fusion concat --emb-dir /emb/xlsr --lomo

Métricas (EER, AUC) em numpy puro. Para multiclasse, EER/AUC usam a redução
bonafide-vs-resto (score de detecção = 1 - P(bonafide)); acurácia e F1-macro
usam as classes completas.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ModelConfig, build_model
from losses import FeatureLoss
from dataset import (
    BRSpeechDataset, NpyEmbeddingLoader, NumpyScaler, collate_fn,
    grouped_split, leave_one_model_out, load_chunks, build_feature_frame, map_labels,
    make_dataloaders, resolve_feature_subset,
)


# --------------------------------------------------------------------------- #
# Métricas (numpy, vetorizadas)
# --------------------------------------------------------------------------- #
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """labels: 1 = spoof (positivo), 0 = bonafide. score alto => mais spoof."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tps = np.cumsum(lab)
    fps = np.cumsum(1 - lab)
    fpr = fps / n_neg            # FAR
    fnr = 1.0 - tps / n_pos      # FRR
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=float)
    i, r = 0, 1
    while i < len(scores):                       # ranks médios para empates
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = (r + r + (j - i)) / 2.0
        r += (j - i + 1)
        i = j + 1
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = ranks_sorted
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_f1(preds: np.ndarray, y: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
def _model_inputs(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items() if k != "label"}


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        out = model(**_model_inputs(batch, device))
        all_logits.append(out["logits"].cpu())
        all_y.append(batch["label"])
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)
    det_score = 1.0 - probs[:, 0]                # P(não-bonafide)
    y_bin = (y != 0).astype(int)
    return {
        "eer": compute_eer(det_score, y_bin),
        "auc": roc_auc(det_score, y_bin),
        "acc": float((preds == y).mean()),
        "f1": macro_f1(preds, y, num_classes),
    }


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights_from_labels(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (num_classes * counts)          # frequência inversa
    return torch.tensor(w, dtype=torch.float32)


def build_optimizer(model, lr: float, backbone_lr: float, weight_decay: float):
    if getattr(model, "ssl_backbone", None) is not None:
        backbone_ids = {id(p) for p in model.ssl_backbone.parameters()}
        groups = [
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": lr},
            {"params": list(model.ssl_backbone.parameters()), "lr": backbone_lr},
        ]
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_run(cfg: ModelConfig, train_loader, val_loader, *,
                  train_labels: np.ndarray, ssl_backbone=None, epochs=30, lr=1e-3,
                  backbone_lr=1e-5, weight_decay=1e-4, use_feature_loss=False,
                  alpha=1.0, beta=1.0, patience=5, device="cpu", verbose=True):
    model = build_model(cfg, ssl_backbone=ssl_backbone).to(device)
    emb_dim = cfg.d_model * (2 if cfg.fusion == "concat" else 1)
    cw = class_weights_from_labels(train_labels, cfg.num_classes).to(device)
    crit = FeatureLoss(
        dim=emb_dim, class_weights=cw,
        alpha=alpha if use_feature_loss else 0.0,
        beta=beta if use_feature_loss else 0.0,
    ).to(device)
    opt = build_optimizer(model, lr, backbone_lr, weight_decay)

    best = {"eer": float("inf")}
    best_state, history, bad = None, [], 0
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            opt.zero_grad()
            out = model(**_model_inputs(batch, device))
            loss, _ = crit(out["logits"], out["embedding"], batch["label"].to(device))
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(batch["label"])
        train_loss = running / len(train_loader.dataset)
        val = evaluate(model, val_loader, device, cfg.num_classes)
        history.append({"epoch": ep, "train_loss": train_loss, **val})
        if verbose:
            print(f"  ep {ep:02d}  loss={train_loss:.4f}  "
                  f"val EER={val['eer']:.4f}  AUC={val['auc']:.4f}  "
                  f"acc={val['acc']:.3f}  f1={val['f1']:.3f}")
        if val["eer"] < best["eer"]:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop (sem melhora em {patience} épocas)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, history


# --------------------------------------------------------------------------- #
# Helper: DataLoaders a partir de índices (usado no LOMO)
# --------------------------------------------------------------------------- #
def loaders_from_indices(frame, feature_names, y, idx_map: dict, scaler, *,
                         batch_size=32, embedding_loader=None, waveform_loader=None,
                         num_workers=0):
    loaders = {}
    for name, idx in idx_map.items():
        ds = BRSpeechDataset(frame.iloc[idx], feature_names, y[idx], scaler,
                             embedding_loader, waveform_loader)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                                   collate_fn=collate_fn, num_workers=num_workers)
    return loaders


def _resolve_io(args):
    """Decide a rota de SSL a partir dos args: waveform (extração em forward) ou
    embeddings .npy pré-extraídos. Devolve (embedding_loader, waveform_loader,
    ssl_backbone, freeze_ssl)."""
    if args.waveform:
        from embeddings import WaveformLoader, build_xlsr_backbone
        freeze = not args.finetune_ssl
        backbone = build_xlsr_backbone(args.ssl_model, freeze=freeze)
        return None, WaveformLoader(), backbone, freeze
    emb = NpyEmbeddingLoader(args.emb_dir) if args.emb_dir else None
    return emb, None, None, True


# --------------------------------------------------------------------------- #
# Leave-one-spoof_model-out
# --------------------------------------------------------------------------- #
def run_lomo(args, device):
    df = load_chunks(args.metrics_path)
    df, feature_names = build_feature_frame(df)
    from dataset import resolve_feature_subset
    feature_names = resolve_feature_subset(feature_names, args.feature_subset)
    y_bin, _ = map_labels(df["model"], multiclass=False, bonafide_aliases=args.bonafide_aliases)

    emb_loader, wav_loader, backbone, freeze = _resolve_io(args)
    rows = []
    for held, tr_all, te in leave_one_model_out(df["model"], args.bonafide_aliases):
        # split interno treino/val por locutor (sem vazamento)
        sub_split = grouped_split(df.iloc[tr_all]["patient"], fracs=(0.85, 0.15, 0.0), seed=args.seed)
        tr = tr_all[sub_split["train"]]
        va = tr_all[sub_split["val"]] if len(sub_split["val"]) else tr_all[sub_split["test"]]

        scaler = NumpyScaler().fit(df.iloc[tr][feature_names].to_numpy(dtype=float))
        loaders = loaders_from_indices(
            df, feature_names, y_bin,
            {"train": tr, "val": va, "test": te}, scaler,
            batch_size=args.batch_size, embedding_loader=emb_loader, waveform_loader=wav_loader,
        )
        cfg = ModelConfig(handcrafted_dim=len(feature_names), fusion=args.fusion,
                          num_classes=2, d_model=args.d_model, freeze_ssl=freeze)
        print(f"\n[LOMO] held-out = {held}  (train={len(tr)} val={len(va)} test={len(te)})")
        model, _, _ = train_one_run(
            cfg, loaders["train"], loaders["val"], train_labels=y_bin[tr], ssl_backbone=backbone,
            epochs=args.epochs, lr=args.lr, use_feature_loss=args.feature_loss,
            patience=args.patience, device=device, verbose=args.verbose,
        )
        test = evaluate(model, loaders["test"], device, 2)
        print(f"[LOMO] {held}: test EER={test['eer']:.4f}  AUC={test['auc']:.4f}")
        rows.append({"held_out_model": held, **test})

    print("\n=== Tabela de generalização (unseen synthesizer) ===")
    hdr = f"{'spoof_model':16s} {'EER':>7s} {'AUC':>7s} {'acc':>7s} {'f1':>7s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['held_out_model']:16s} {r['eer']:7.4f} {r['auc']:7.4f} "
              f"{r['acc']:7.3f} {r['f1']:7.3f}")
    eers = [r["eer"] for r in rows if not np.isnan(r["eer"])]
    if eers:
        print("-" * len(hdr))
        print(f"{'média':16s} {np.mean(eers):7.4f}")
    return rows


# --------------------------------------------------------------------------- #
# Treino padrão (split agrupado)
# --------------------------------------------------------------------------- #
def run_standard(args, device):
    emb_loader, wav_loader, backbone, freeze = _resolve_io(args)
    bundle = make_dataloaders(
        args.metrics_path, multiclass=args.multiclass, batch_size=args.batch_size,
        seed=args.seed, embedding_loader=emb_loader, waveform_loader=wav_loader,
        bonafide_aliases=args.bonafide_aliases,
        feature_subset=args.feature_subset,
    )
    num_classes = len(bundle.label_info.classes)
    cfg = ModelConfig(handcrafted_dim=len(bundle.feature_names), fusion=args.fusion,
                      num_classes=num_classes, d_model=args.d_model, freeze_ssl=freeze)
    print(f"features={len(bundle.feature_names)}  subset={args.feature_subset}  "
          f"classes={bundle.label_info.classes}  fusion={args.fusion}  "
          f"feature_names={bundle.feature_names}")
    print(f"split: {{ {', '.join(f'{k}={len(v.dataset)}' for k, v in bundle.loaders.items())} }}")

    train_y = bundle.frame.iloc[bundle.splits["train"]]["model"]
    train_labels, _ = map_labels(train_y, multiclass=args.multiclass,
                                 bonafide_aliases=args.bonafide_aliases)

    model, best_val, _ = train_one_run(
        cfg, bundle.loaders["train"], bundle.loaders["val"], train_labels=train_labels,
        ssl_backbone=backbone, epochs=args.epochs, lr=args.lr,
        use_feature_loss=args.feature_loss, patience=args.patience,
        device=device, verbose=args.verbose,
    )
    test = evaluate(model, bundle.loaders["test"], device, num_classes)
    print(f"\nMelhor val: EER={best_val['eer']:.4f} AUC={best_val['auc']:.4f}")
    print(f"Teste:      EER={test['eer']:.4f} AUC={test['auc']:.4f} "
          f"acc={test['acc']:.3f} f1={test['f1']:.3f}")

    if args.out:
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "feature_names": bundle.feature_names,
            "scaler": {"mean": bundle.scaler.mean_, "std": bundle.scaler.std_},
            "classes": bundle.label_info.classes,
            "test_metrics": test,
        }, args.out)
        print(f"checkpoint salvo em {args.out}")
    return test


# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Treino do detector de deepfake (BRSpeech-DF)")
    p.add_argument("metrics_path", type=str, help="parquet, diretório de chunks ou glob")
    p.add_argument("--fusion", default="concat",
                   choices=["wav2vec_only", "handcrafted_only", "concat", "cross_attention"])
    p.add_argument("--multiclass", action="store_true", help="classifica por spoof_model")
    p.add_argument("--emb-dir", type=str, default=None, help="dir de .npy com embeddings XLS-R")
    p.add_argument("--waveform", action="store_true",
                   help="extrai o XLS-R DENTRO do forward a partir do waveform")
    p.add_argument("--ssl-model", type=str, default="facebook/wav2vec2-xls-r-300m",
                   help="backbone HuggingFace p/ --waveform")
    p.add_argument("--finetune-ssl", action="store_true",
                   help="treina o backbone junto (default: congelado)")
    p.add_argument("--lomo", action="store_true", help="roda leave-one-spoof_model-out")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--feature-loss", action="store_true", help="ativa center+contrast loss")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bonafide-aliases", nargs="+", default=["bonafide", "real", "genuine", "human", "bona"])
    p.add_argument("--out", type=str, default=None, help="caminho p/ salvar checkpoint")
    p.add_argument(
        "--feature-subset", type=str, default="all",
        metavar="SUBSET",
        help=(
            "Subconjunto de features handcrafted para ablação. "
            "Valores: all | scalar | pause | f0 | combinações com vírgula "
            "(ex: scalar,pause). Default: all."
        ),
    )
    p.add_argument("--quiet", dest="verbose", action="store_false")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    t0 = time.time()
    if args.lomo:
        run_lomo(args, device)
    else:
        run_standard(args, device)
    print(f"tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()