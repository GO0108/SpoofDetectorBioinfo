"""
dataset.py — Consome os chunks de métricas do BRSpeech-DF e entrega tensores
no formato esperado por model.py (DeepfakeFusionModel).

Pipeline:
    1. carrega/concatena todos os parquets de métricas (chunks)
    2. agrega o contorno de F0 do REAPER em escalares (mascarando frames -1)
    3. monta o vetor handcrafted, dropando colineares exatos (ddp = 3*rap, dda = 3*apq3)
    4. divide por `speaker` (split agrupado, sem vazamento de locutor)
    5. mapeia o rótulo a partir da coluna `model` (bonafide -> 0, spoof -> >=1),
       de forma consistente entre os 3 splits
    6. padroniza as features ajustando o scaler SÓ no treino
    7. (opcional) anexa embeddings XLS-R por utterance para a fusão

Cache de features por split (train/dev/test):
    Os passos 1-4 (carregar chunks, agregar F0, montar handcrafted, dividir
    por locutor) são determinísticos dado (metrics_path, seed, fracs) e podem
    ser caros de repetir — é exatamente o que acontece numa ablação como a do
    run_experiments.sh, que chama Train.py ~15 vezes sobre o mesmo parquet.

    Passando `feature_cache_dir=...` para make_dataloaders():
      - Se os 3 arquivos abaixo já existirem, a extração é PULADA e eles são
        lidos diretamente (apenas a leitura do parquet, sem recomputar nada):
            <feature_cache_dir>/train_features.parquet
            <feature_cache_dir>/dev_features.parquet
            <feature_cache_dir>/test_features.parquet
      - Caso contrário, a extração roda normalmente e o resultado de cada
        split é salvo nesses caminhos para a próxima chamada reusar.
    A checagem é só de existência dos arquivos (não recalcula/valida
    conteúdo); se mudar `seed`/`fracs`/o parquet de entrada, apague o
    diretório de cache (ou aponte para outro) para forçar uma nova extração.

Schema esperado (de extract_metrics.py):
    file, path, split, model, speaker, times_reaper, f0_reaper, corr,
    jitter_local_pct, shimmer_local_pct, hnr_mean_dB,
    rms_mean, zcr_mean, n_pausas, pause_dur_mean
    (+ demais colunas do parquet ignoradas pelo loader)
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# Escalares mantidos: apenas as features selecionadas para fusão com XLS-R.
# Variantes redundantes de jitter/shimmer (rap, ppq5, ddp, dB, apq3, apq5, dda)
# foram removidas; somente local é mantido para cada um.
SCALAR_FEATURES: list[str] = [
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_mean_dB",
]
# Features de pausa derivadas de frames unvoiced (f0 < 0 no REAPER).
# Ressalva: incluem consoantes surdas além de silêncio puro — ver notebook de análise.
PAUSE_FEATURES: list[str] = [
    "rms_mean",
    "zcr_mean",
    "n_pausas",
    "pause_dur_mean",
]
# Agregados de F0 mantidos: média e desvio (os mais discriminativos segundo KL).
# f0_median, f0_range, voiced_fraction e corr_mean foram descartados.
F0_AGG_FEATURES: list[str] = [
    "f0_mean",
    "f0_std",
]
# Grupos para ablação via --feature-subset
# Chaves: "scalar" | "pause" | "f0" | combinações com vírgula | "all"
FEATURE_GROUPS: dict[str, list[str]] = {
    "scalar": ["jitter_local_pct", "shimmer_local_pct", "hnr_mean_dB"],
    "pause":  ["rms_mean", "zcr_mean", "n_pausas", "pause_dur_mean"],
    "f0":     ["f0_mean", "f0_std"],
}

BONAFIDE_ALIASES: tuple[str, ...] = ("bonafide", "real", "genuine", "human", "bona")

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Cache de features por split (train/dev/test) — ver docstring do módulo
# --------------------------------------------------------------------------- #
# Nome de arquivo no cache para cada split interno. "val" é salvo como "dev"
# (sinônimo comum em protocolos de anti-spoofing: train / dev / eval).
FEATURE_CACHE_FILENAMES: dict[str, str] = {
    "train": "train_features.parquet",
    "val":   "dev_features.parquet",
    "test":  "test_features.parquet",
}

# Colunas de contorno (arrays) já resumidas em escalares por build_feature_frame;
# não precisam ir para o cache e nem sempre são serializáveis em parquet simples.
_ARRAY_COLUMNS_TO_DROP: tuple[str, ...] = ("times_reaper", "f0_reaper", "corr")


def feature_cache_paths(cache_dir: str | Path) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    return {name: cache_dir / fname for name, fname in FEATURE_CACHE_FILENAMES.items()}


def _cache_meta_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "meta.json"


def load_feature_cache(
    cache_dir: str | Path,
    seed: Optional[int] = None,
    fracs: Optional[Sequence[float]] = None,
) -> Optional[dict[str, pd.DataFrame]]:
    """Devolve {"train","val","test": DataFrame} se os 3 parquets já existirem
    em `cache_dir`; devolve None se o cache estiver ausente ou incompleto
    (sinal para o chamador rodar a extração normalmente).

    A checagem é SÓ de existência dos arquivos — não recalcula nada. Se
    `seed`/`fracs` forem passados e diferirem do que está em meta.json (salvo
    junto do cache), apenas um aviso é impresso; o cache é usado do mesmo jeito.
    """
    paths = feature_cache_paths(cache_dir)
    if not all(p.exists() for p in paths.values()):
        return None

    meta_path = _cache_meta_path(cache_dir)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        if seed is not None and meta.get("seed") not in (None, seed):
            print(f"[cache][AVISO] seed do cache ({meta.get('seed')}) difere do "
                  f"seed atual ({seed}); usando o cache existente mesmo assim "
                  f"(apague {cache_dir} para forçar nova extração).")
        if fracs is not None and meta.get("fracs") not in (None, list(fracs)):
            print(f"[cache][AVISO] fracs do cache ({meta.get('fracs')}) difere do "
                  f"atual ({list(fracs)}); usando o cache existente mesmo assim "
                  f"(apague {cache_dir} para forçar nova extração).")

    return {name: pd.read_parquet(p) for name, p in paths.items()}


def save_feature_cache(
    cache_dir: str | Path,
    split_frames: dict[str, pd.DataFrame],
    seed: Optional[int] = None,
    fracs: Optional[Sequence[float]] = None,
) -> None:
    """Salva cada split (já com as features handcrafted construídas) em
    <cache_dir>/{train,dev,test}_features.parquet + meta.json (seed/fracs)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = feature_cache_paths(cache_dir)
    for name, frame in split_frames.items():
        drop = [c for c in _ARRAY_COLUMNS_TO_DROP if c in frame.columns]
        frame.drop(columns=drop).to_parquet(paths[name], index=False)
    _cache_meta_path(cache_dir).write_text(
        json.dumps({"seed": seed, "fracs": list(fracs) if fracs is not None else None})
    )


# --------------------------------------------------------------------------- #
# Carregamento dos chunks
# --------------------------------------------------------------------------- #
def load_chunks(path: str | Path) -> pd.DataFrame:
    """Aceita um arquivo .parquet, um diretório (lê *.parquet) ou um glob."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
    elif any(ch in str(path) for ch in "*?["):
        files = sorted(Path(f) for f in glob.glob(str(path)))
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"Nenhum parquet em: {path}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df


# --------------------------------------------------------------------------- #
# Agregação do F0 (contorno -> escalares)
# --------------------------------------------------------------------------- #
def aggregate_f0_row(f0_reaper, corr) -> dict:
    f0 = np.asarray(f0_reaper, dtype=float)
    voiced = f0 > 0                                   # REAPER marca não-vozeado como -1
    v = f0[voiced]
    if v.size == 0:
        return {k: np.nan for k in F0_AGG_FEATURES}
    return {
        "f0_mean": float(v.mean()),
        "f0_std":  float(v.std()),
    }


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Garante todas as colunas handcrafted e monta a lista de features na ordem canônica.

    Ordem das 9 features para fusão com XLS-R:
        jitter_local_pct, shimmer_local_pct, hnr_mean_dB  (SCALAR_FEATURES)
        rms_mean, zcr_mean, n_pausas, pause_dur_mean       (PAUSE_FEATURES)
        f0_mean, f0_std                                    (F0_AGG_FEATURES)

    - Agrega F0 a partir do contorno REAPER se a coluna 'f0_reaper' existir;
      caso contrário, assume que 'f0_mean'/'f0_std' já estão no parquet.
    - Features de pausa devem estar no parquet (produzidas pela seção 4 do
      notebook de análise). Se ausentes, levanta KeyError com instrução.
    """
    df = df.copy()

    # Agrega F0 se o contorno estiver disponível
    if "f0_reaper" in df.columns:
        aggs = [aggregate_f0_row(r.f0_reaper, getattr(r, "corr", None))
                for r in df.itertuples(index=False)]
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(aggs)], axis=1)

    # Valida features obrigatórias
    required = set(SCALAR_FEATURES) | set(PAUSE_FEATURES) | set(F0_AGG_FEATURES)
    missing = required - set(df.columns)
    if missing:
        pause_missing = missing & set(PAUSE_FEATURES)
        if pause_missing:
            raise KeyError(
                f"Features de pausa ausentes no parquet: {sorted(pause_missing)}. "
                "Certifique-se de ter rodado a extração de pausas do notebook de análise "
                "(seção 4) e salvo as colunas rms_mean, zcr_mean, n_pausas e pause_dur_mean."
            )
        raise KeyError(f"Colunas de feature ausentes no parquet: {sorted(missing)}")

    # Ordem canônica: scalar -> pausa -> f0
    all_features = SCALAR_FEATURES + PAUSE_FEATURES + F0_AGG_FEATURES
    feature_names = [c for c in all_features if c in df.columns]
    return df, feature_names


# --------------------------------------------------------------------------- #
# Rótulos
# --------------------------------------------------------------------------- #
@dataclass
class LabelInfo:
    classes: list[str]
    mapping: dict[str, int]      # valor de `model` -> id de classe
    multiclass: bool


def map_labels(models: pd.Series, multiclass: bool = False,
               bonafide_aliases: Sequence[str] = BONAFIDE_ALIASES) -> tuple[np.ndarray, LabelInfo]:
    aliases = {a.lower() for a in bonafide_aliases}
    is_bona = models.astype(str).str.lower().isin(aliases)
    if not is_bona.any():
        raise ValueError(
            f"Nenhum valor de `model` casou com bonafide {sorted(aliases)}. "
            f"Valores encontrados: {sorted(models.unique())[:10]}. "
            "Ajuste bonafide_aliases."
        )
    if not multiclass:
        y = (~is_bona).astype(int).to_numpy()
        return y, LabelInfo(["bonafide", "spoof"], {}, False)

    spoof_models = sorted(models[~is_bona].astype(str).unique())
    mapping = {m: i + 1 for i, m in enumerate(spoof_models)}
    classes = ["bonafide"] + spoof_models
    y = np.where(is_bona, 0, models.astype(str).map(mapping)).astype(int)
    return y, LabelInfo(classes, mapping, True)


# --------------------------------------------------------------------------- #
# Scaler com tratamento de NaN (serializável, sem sklearn)
# --------------------------------------------------------------------------- #
@dataclass
class NumpyScaler:
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "NumpyScaler":
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), self.mean_, X)     # imputa pela média do treino
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Split agrupado por locutor
# --------------------------------------------------------------------------- #
def grouped_split(groups: pd.Series, fracs=(0.7, 0.15, 0.15), seed: int = 0) -> dict:
    """Divide mantendo cada valor de `groups` (ex.: `speaker`) inteiro num único split."""
    assert abs(sum(fracs) - 1.0) < 1e-6, "fracs deve somar 1"
    uniq = np.array(groups.astype(str).unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    buckets = {"train": set(uniq[:n_tr]),
               "val": set(uniq[n_tr:n_tr + n_va]),
               "test": set(uniq[n_tr + n_va:])}
    g = groups.astype(str).to_numpy()
    return {k: np.where(np.isin(g, list(v)))[0] for k, v in buckets.items()}


def leave_one_model_out(models: pd.Series, bonafide_aliases: Sequence[str] = BONAFIDE_ALIASES):
    """Itera (held_out_model, train_idx, test_idx): treina sem um spoof_model
    (bonafide permanece no treino) e testa nele + bonafide. Protocolo
    'unseen synthesizer' do ASVspoof DF."""
    aliases = {a.lower() for a in bonafide_aliases}
    m = models.astype(str)
    is_bona = m.str.lower().isin(aliases).to_numpy()
    for held in sorted(m[~is_bona].unique()):
        held_mask = (m == held).to_numpy()
        train_idx = np.where(~held_mask)[0]            # tudo menos o modelo retido
        test_idx = np.where(held_mask | is_bona)[0]    # modelo retido + bonafide
        yield held, train_idx, test_idx


# --------------------------------------------------------------------------- #
# Loaders de embedding XLS-R (opcional)
# --------------------------------------------------------------------------- #
class NpyEmbeddingLoader:
    """Carrega embeddings (T, D) de arquivos .npy nomeados pelo stem do áudio.
    Ex.: 'spk1_0001.flac' -> <emb_dir>/spk1_0001.npy"""

    def __init__(self, emb_dir: str | Path, key: str = "file"):
        self.emb_dir = Path(emb_dir)
        self.key = key

    def __call__(self, row: pd.Series) -> Optional[np.ndarray]:
        stem = Path(str(row[self.key])).stem
        f = self.emb_dir / f"{stem}.npy"
        return np.load(f) if f.exists() else None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BRSpeechDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_names: list[str], labels: np.ndarray,
                 scaler: NumpyScaler,
                 embedding_loader: Optional[Callable[[pd.Series], Optional[np.ndarray]]] = None,
                 waveform_loader: Optional[Callable[[pd.Series], np.ndarray]] = None):
        assert not (embedding_loader and waveform_loader), \
            "use embedding_loader OU waveform_loader, não os dois"
        self.frame = frame.reset_index(drop=True)
        self.feature_names = feature_names
        X = self.frame[feature_names].to_numpy(dtype=float)
        self.X = scaler.transform(X).astype(np.float32)
        self.y = labels.astype(np.int64)
        self.embedding_loader = embedding_loader
        self.waveform_loader = waveform_loader

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> dict:
        item = {
            "handcrafted": torch.from_numpy(self.X[i]),
            "label": torch.tensor(self.y[i], dtype=torch.long),
        }
        if self.embedding_loader is not None:
            emb = self.embedding_loader(self.frame.iloc[i])
            if emb is not None:
                item["ssl_features"] = torch.as_tensor(np.asarray(emb), dtype=torch.float32)
        if self.waveform_loader is not None:
            wav = self.waveform_loader(self.frame.iloc[i])
            item["waveform"] = torch.as_tensor(np.asarray(wav), dtype=torch.float32)
        return item


def collate_fn(batch: list[dict]) -> dict:
    """Empilha handcrafted/labels; faz pad das sequências SSL e dos waveforms."""
    out = {
        "handcrafted": torch.stack([b["handcrafted"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    if "ssl_features" in batch[0]:
        seqs = [b["ssl_features"] for b in batch]
        T = max(s.shape[0] for s in seqs); D = seqs[0].shape[1]
        feats = torch.zeros(len(seqs), T, D, dtype=torch.float32)
        mask = torch.zeros(len(seqs), T, dtype=torch.bool)
        for j, s in enumerate(seqs):
            feats[j, : s.shape[0]] = s
            mask[j, : s.shape[0]] = True
        out["ssl_features"], out["ssl_mask"] = feats, mask
    if "waveform" in batch[0]:
        wavs = [b["waveform"] for b in batch]
        N = max(w.shape[0] for w in wavs)
        buf = torch.zeros(len(wavs), N, dtype=torch.float32)
        wmask = torch.zeros(len(wavs), N, dtype=torch.bool)
        for j, w in enumerate(wavs):
            buf[j, : w.shape[0]] = w
            wmask[j, : w.shape[0]] = True
        out["waveform"], out["waveform_mask"] = buf, wmask
    return out


# --------------------------------------------------------------------------- #
# Conveniência: do parquet aos DataLoaders
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    loaders: dict[str, DataLoader]
    scaler: NumpyScaler
    feature_names: list[str]
    label_info: LabelInfo
    frame: pd.DataFrame = field(repr=False)
    splits: dict = field(repr=False, default_factory=dict)


def resolve_feature_subset(feature_names: list[str], subset: str) -> list[str]:
    """Filtra feature_names para o subconjunto pedido.

    subset pode ser:
        "all"                → todas as features (sem filtro)
        "scalar"             → só FEATURE_GROUPS["scalar"]
        "pause"              → só FEATURE_GROUPS["pause"]
        "f0"                 → só FEATURE_GROUPS["f0"]
        "scalar,pause"       → união de dois grupos (qualquer combinação)
        "scalar,pause,f0"    → equivalente a "all"

    Apenas features que efetivamente existem em feature_names são retornadas
    (evita KeyError se uma coluna estiver ausente no parquet).
    """
    if subset == "all":
        return feature_names
    requested: list[str] = []
    for group_key in subset.split(","):
        group_key = group_key.strip()
        if group_key not in FEATURE_GROUPS:
            raise ValueError(
                f"Subconjunto de feature desconhecido: '{group_key}'. "
                f"Válidos: {sorted(FEATURE_GROUPS)} ou 'all'."
            )
        requested.extend(FEATURE_GROUPS[group_key])
    # Mantém a ordem canônica e só o que existe no parquet
    return [f for f in feature_names if f in requested]


def make_dataloaders(
    metrics_path: str | Path,
    multiclass: bool = False,
    batch_size: int = 32,
    fracs=(0.7, 0.15, 0.15),
    seed: int = 0,
    embedding_loader: Optional[Callable] = None,
    waveform_loader: Optional[Callable] = None,
    bonafide_aliases: Sequence[str] = BONAFIDE_ALIASES,
    num_workers: int = 0,
    feature_subset: str = "all",
    feature_cache_dir: Optional[str | Path] = None,
) -> DataBundle:
    """
    feature_cache_dir:
        Se informado, faz cache em disco do split train/dev/test já com as
        features handcrafted construídas (pós build_feature_frame +
        grouped_split). Se os 3 parquets já existirem nesse diretório, a
        extração (load_chunks + build_feature_frame + grouped_split) é
        PULADA e eles são lidos diretamente; caso contrário, a extração roda
        normalmente e o resultado é salvo lá para a próxima chamada. Ver
        docstring do módulo para os nomes de arquivo. `--feature-subset` é
        aplicado depois da leitura do cache, então o mesmo cache serve para
        qualquer ablação de subconjunto de features.
    """
    cached = load_feature_cache(feature_cache_dir, seed=seed, fracs=fracs) if feature_cache_dir else None

    if cached is not None:
        print(f"[cache] features já extraídas encontradas em '{feature_cache_dir}' "
              "— pulando extração e usando o cache.")
        split_frames = cached
        feature_names_all = [
            c for c in (SCALAR_FEATURES + PAUSE_FEATURES + F0_AGG_FEATURES)
            if c in split_frames["train"].columns
        ]
    else:
        df = load_chunks(metrics_path)
        df, feature_names_all = build_feature_frame(df)
        idx_splits = grouped_split(df["speaker"], fracs, seed)
        split_frames = {
            name: df.iloc[idx].reset_index(drop=True) for name, idx in idx_splits.items()
        }
        if feature_cache_dir:
            save_feature_cache(feature_cache_dir, split_frames, seed=seed, fracs=fracs)
            print(f"[cache] features extraídas e salvas em '{feature_cache_dir}' "
                  "(próximas chamadas vão pular a extração).")

    feature_names = resolve_feature_subset(feature_names_all, feature_subset)
    if not feature_names:
        raise ValueError(
            f"Nenhuma feature disponível para subset='{feature_subset}'. "
            "Verifique se as colunas existem no parquet/cache."
        )

    # Mapeia rótulos de forma CONSISTENTE entre os 3 splits — importante no
    # multiclasse, onde o id de cada spoof_model não pode depender de quais
    # splits foram vistos primeiro (ex.: ao ler do cache, cada split vem de um
    # parquet separado em vez de um único df pré-split).
    combined_models = pd.concat(
        [split_frames[n]["model"] for n in SPLIT_NAMES], ignore_index=True
    )
    y_combined, label_info = map_labels(combined_models, multiclass, bonafide_aliases)
    lens = {n: len(split_frames[n]) for n in SPLIT_NAMES}
    y_splits, offset = {}, 0
    for n in SPLIT_NAMES:
        y_splits[n] = y_combined[offset: offset + lens[n]]
        offset += lens[n]

    scaler = NumpyScaler().fit(split_frames["train"][feature_names].to_numpy(dtype=float))

    loaders = {}
    for name in SPLIT_NAMES:
        ds = BRSpeechDataset(split_frames[name], feature_names, y_splits[name], scaler,
                             embedding_loader, waveform_loader)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            collate_fn=collate_fn, num_workers=num_workers,
        )

    # Mantém DataBundle.frame/.splits no mesmo formato de antes (um único
    # frame concatenado + índices posicionais por split), para não quebrar
    # quem já consome esses dois atributos (ex.: Train.py, smoke tests).
    combined_frame = pd.concat([split_frames[n] for n in SPLIT_NAMES], ignore_index=True)
    splits_idx, offset = {}, 0
    for n in SPLIT_NAMES:
        splits_idx[n] = np.arange(offset, offset + lens[n])
        offset += lens[n]

    return DataBundle(loaders, scaler, feature_names, label_info, combined_frame, splits_idx)


# --------------------------------------------------------------------------- #
# Smoke test: gera parquets sintéticos no schema do extract_metrics e roda tudo.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    models = ["bonafide", "tts_a", "vc_b"]
    rows = []
    for spk in range(8):                                   # 8 locutores
        speaker = f"spk{spk:02d}"
        for model in models:
            for _ in range(rng.integers(3, 6)):
                n = int(rng.integers(80, 160))
                f0 = rng.normal(150, 30, n)
                f0[rng.random(n) < 0.4] = -1.0             # frames não-vozeados
                rows.append({
                    "file": f"{speaker}_{model}_{rng.integers(1e6)}.flac",
                    "path": f"/data/{model}/{speaker}/x.flac",
                    "model": model, "speaker": speaker,
                    # contorno de F0 (agregado em build_feature_frame)
                    "f0_reaper": f0, "corr": np.clip(rng.normal(0.7, 0.1, n), 0, 1),
                    # SCALAR_FEATURES
                    "jitter_local_pct":  rng.normal(0.5, 0.2),
                    "shimmer_local_pct": rng.normal(3.0, 1.0),
                    "hnr_mean_dB":       rng.normal(18.0, 3.0),
                    # PAUSE_FEATURES (normalmente produzidas pelo notebook seção 4)
                    "rms_mean":       abs(rng.normal(0.01, 0.005)),
                    "zcr_mean":       abs(rng.normal(0.08, 0.02)),
                    "n_pausas":       int(rng.integers(10, 40)),
                    "pause_dur_mean": abs(rng.normal(0.18, 0.06)),
                })
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as d:
        # salva em 2 chunks, como no fluxo real
        df.iloc[: len(df)//2].to_parquet(Path(d) / "chunk_00.parquet", index=False)
        df.iloc[len(df)//2:].to_parquet(Path(d) / "chunk_01.parquet", index=False)

        bundle = make_dataloaders(d, multiclass=True, batch_size=16, seed=1)
        print("features:", bundle.feature_names)
        print("handcrafted_dim =", len(bundle.feature_names))
        print("classes:", bundle.label_info.classes)
        print("tamanhos:", {k: len(v.dataset) for k, v in bundle.loaders.items()})

        # sem vazamento de locutor entre splits
        spk_sets = {k: set(bundle.frame.iloc[ix]["speaker"]) for k, ix in bundle.splits.items()}
        overlap = (spk_sets["train"] & spk_sets["val"]) | (spk_sets["train"] & spk_sets["test"]) | (spk_sets["val"] & spk_sets["test"])
        print("locutores sobrepostos entre splits:", overlap or "nenhum")

        batch = next(iter(bundle.loaders["train"]))
        print("batch handcrafted:", tuple(batch["handcrafted"].shape),
              "| label:", tuple(batch["label"].shape))

        # leave-one-model-out
        for held, tr, te in leave_one_model_out(bundle.frame["model"]):
            print(f"LOMO held={held:8s} train={len(tr):3d} test={len(te):3d}")

        # --- cache de features por split (train/dev/test) -------------------- #
        cache_dir = Path(d) / "feature_cache"
        print(f"\n[teste cache] diretório: {cache_dir}  (não existe ainda)")
        b1 = make_dataloaders(d, multiclass=True, batch_size=16, seed=1,
                              feature_cache_dir=cache_dir)
        assert feature_cache_paths(cache_dir)["train"].exists(), "cache de treino não foi salvo"
        assert feature_cache_paths(cache_dir)["val"].exists(), "cache de dev não foi salvo"
        assert feature_cache_paths(cache_dir)["test"].exists(), "cache de test não foi salvo"

        print(f"[teste cache] chamando de novo com o MESMO cache_dir — deve pular a extração")
        b2 = make_dataloaders(d, multiclass=True, batch_size=16, seed=1,
                              feature_cache_dir=cache_dir)
        assert b1.feature_names == b2.feature_names
        assert b1.label_info.classes == b2.label_info.classes
        assert {k: len(v.dataset) for k, v in b1.loaders.items()} == \
               {k: len(v.dataset) for k, v in b2.loaders.items()}
        assert np.allclose(b1.scaler.mean_, b2.scaler.mean_, equal_nan=True)
        print("[teste cache] OK — segunda chamada usou o cache e bateu com a primeira")

        # --feature-subset diferente deve reusar o MESMO cache (filtro é pós-cache)
        b3 = make_dataloaders(d, multiclass=True, batch_size=16, seed=1,
                              feature_cache_dir=cache_dir, feature_subset="pause")
        print(f"[teste cache] subset='pause' sobre o mesmo cache -> features={b3.feature_names}")
        assert b3.feature_names == FEATURE_GROUPS["pause"]
    print("OK")