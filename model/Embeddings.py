"""
embeddings.py — Extração DINÂMICA de embeddings XLS-R (sob demanda, com cache).

Em vez de despejar todos os embeddings em disco antes de treinar, o
`DynamicEmbeddingLoader` calcula o embedding na primeira vez que o arquivo
aparece e reusa depois (cache em memória e/ou em disco). Plugue-o no slot
`embedding_loader` de dataset.make_dataloaders — nada mais muda.

    from embeddings import XLSRExtractor, DynamicEmbeddingLoader
    extractor = XLSRExtractor("facebook/wav2vec2-xls-r-300m")
    loader = DynamicEmbeddingLoader(extractor, cache_dir="emb_cache")
    bundle = make_dataloaders(chunks, fusion="concat", embedding_loader=loader)

IMPORTANTE sobre paralelismo:
    O extractor roda um modelo grande dentro do __getitem__. Use num_workers=0
    enquanto o cache está "frio" (ou se o extractor estiver na GPU — CUDA não
    sobrevive a fork de workers). Estratégia recomendada: rode uma época para
    aquecer o cache em disco e, das próximas em diante, troque para
    NpyEmbeddingLoader(cache_dir) com num_workers>0 (lê só .npy, rápido).

Fine-tuning do XLS-R: aí o congelamento não serve; passe o backbone e o
waveform ao DeepfakeFusionModel (rota ssl_backbone + waveform), que treina
o extractor no mesmo grafo.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Áudio
# --------------------------------------------------------------------------- #
def _resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    try:
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(y), sr, target).numpy().astype("float32")
    except Exception:
        try:
            from scipy.signal import resample_poly
            g = gcd(sr, target)
            return resample_poly(y, target // g, sr // g).astype("float32")
        except Exception as e:
            raise RuntimeError(
                "Resample requer torchaudio ou scipy; ou forneça áudio já em 16 kHz."
            ) from e


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1).astype("float32")
    y = _resample(y, sr, target_sr)
    return y, target_sr


# --------------------------------------------------------------------------- #
# Backbone p/ extração DENTRO do forward do model.py (rota waveform)
# --------------------------------------------------------------------------- #
def build_xlsr_backbone(model_name: str = "facebook/wav2vec2-xls-r-300m",
                        freeze: bool = True):
    """Devolve um Wav2Vec2Model (HuggingFace) para injetar em build_model(cfg, ssl_backbone=...).
    O DeepfakeFusionModel faz a normalização e a máscara internamente; aqui só
    carregamos os pesos. `transformers` é importado de forma preguiçosa."""
    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(model_name)
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


class WaveformLoader:
    """Devolve o waveform cru (16 kHz, mono) para o dataset; a extração XLS-R
    acontece no forward do modelo. Use com dataset(waveform_loader=...)."""

    def __init__(self, path_col: str = "path", target_sr: int = 16000):
        self.path_col = path_col
        self.target_sr = target_sr

    def __call__(self, row) -> np.ndarray:
        y, _ = load_audio(str(row[self.path_col]), self.target_sr)
        return y


# --------------------------------------------------------------------------- #
# Extrator XLS-R (backbone congelado por padrão)
# --------------------------------------------------------------------------- #
class XLSRExtractor:
    """Wrapper do Wav2Vec2/XLS-R (HuggingFace). `transformers` é importado de
    forma preguiçosa — `import embeddings` funciona sem ele instalado.

    layers:
        "last" -> última hidden state, (T, D)   [casa com ssl_features]
        "all"  -> todas as camadas, (L, T, D)    [para SLS via ssl_hidden_states]
        int    -> uma camada específica, (T, D)
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m",
                 device: Optional[str] = None, layers="last", freeze: bool = True):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.layers = layers
        self.frozen = freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        inputs = self.fe(waveform, sampling_rate=sr, return_tensors="pt")
        iv = inputs.input_values.to(self.device)
        want_hidden = self.layers != "last"
        out = self.model(iv, output_hidden_states=want_hidden)
        if self.layers == "last":
            feat = out.last_hidden_state[0]                       # (T, D)
        elif self.layers == "all":
            feat = torch.stack(out.hidden_states, dim=0)[:, 0]    # (L, T, D)
        else:
            feat = out.hidden_states[int(self.layers)][0]         # (T, D)
        return feat.detach().cpu().numpy().astype("float32")


# --------------------------------------------------------------------------- #
# Loader dinâmico com cache (drop-in para dataset.embedding_loader)
# --------------------------------------------------------------------------- #
class DynamicEmbeddingLoader:
    def __init__(self, extractor, path_col: str = "path", key_col: str = "file",
                 cache_dir: Optional[str] = None, in_memory: bool = False,
                 target_sr: int = 16000):
        self.extractor = extractor
        self.path_col = path_col
        self.key_col = key_col
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mem: Optional[dict] = {} if in_memory else None
        self.target_sr = target_sr

    def _key(self, row) -> str:
        return Path(str(row[self.key_col])).stem

    def __call__(self, row) -> np.ndarray:
        k = self._key(row)
        if self.mem is not None and k in self.mem:
            return self.mem[k]
        if self.cache_dir is not None:
            f = self.cache_dir / f"{k}.npy"
            if f.exists():
                arr = np.load(f)
                if self.mem is not None:
                    self.mem[k] = arr
                return arr
        y, sr = load_audio(str(row[self.path_col]), self.target_sr)
        arr = self.extractor.extract(y, sr)
        if self.cache_dir is not None:
            np.save(self.cache_dir / f"{k}.npy", arr)
        if self.mem is not None:
            self.mem[k] = arr
        return arr


def warm_cache(loader: "DynamicEmbeddingLoader", frame, verbose: bool = True) -> None:
    """Percorre o frame uma vez para popular o cache em disco. Depois disso dá
    para treinar com NpyEmbeddingLoader(cache_dir) e num_workers>0."""
    from tqdm import tqdm
    it = frame.itertuples(index=False)
    if verbose:
        it = tqdm(it, total=len(frame), unit="wav")
    for row in it:
        loader(row._asdict() if hasattr(row, "_asdict") else row)


# --------------------------------------------------------------------------- #
# Smoke test: valida cache/interface com extrator de mentira (sem transformers).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import pandas as pd

    class FakeExtractor:
        dim = 1024
        def extract(self, waveform, sr):
            T = max(1, len(waveform) // 320)            # ~ frame rate do XLS-R
            return np.random.RandomState(len(waveform)).randn(T, self.dim).astype("float32")

    # evita dependência de áudio real: simula a leitura (rebinda o global do módulo)
    load_audio = lambda path, target_sr=16000: (np.zeros(8000, dtype="float32"), target_sr)

    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "cache"
        loader = DynamicEmbeddingLoader(FakeExtractor(), cache_dir=str(cache), in_memory=True)
        row = {"file": "spk1_0001.flac", "path": "/x/spk1_0001.flac"}

        a = loader(row)                                  # miss -> extrai -> grava
        assert (cache / "spk1_0001.npy").exists(), "cache em disco não gravou"
        b = loader(row)                                  # hit (memória)
        assert np.array_equal(a, b)

        loader2 = DynamicEmbeddingLoader(FakeExtractor(), cache_dir=str(cache))
        c = loader2(row)                                 # hit (disco), extractor não chamado
        assert np.array_equal(a, c)
        print(f"embedding shape={a.shape}  cache_miss->disk->hit OK")

        # integração com dataset.py + model.py (fusão concat usando o loader dinâmico)
        import dataset as DS, model as M
        rng = np.random.default_rng(0)
        rows = []
        for spk in range(6):
            for mdl in ["bonafide", "tts_a"]:
                for _ in range(3):
                    n = int(rng.integers(80, 120)); f0 = rng.normal(150, 30, n); f0[rng.random(n) < .4] = -1
                    rows.append(dict(file=f"{spk}_{mdl}_{rng.integers(1_000_000)}.flac", path="/x/a.flac",
                        model=mdl, patient=f"spk{spk}", f0_reaper=f0, corr=np.clip(rng.normal(.7, .1, n), 0, 1),
                        # SCALAR_FEATURES
                        jitter_local_pct=rng.normal(.5, .2), shimmer_local_pct=3., hnr_mean_dB=18.,
                        # PAUSE_FEATURES
                        rms_mean=abs(rng.normal(.01, .005)), zcr_mean=abs(rng.normal(.08, .02)),
                        n_pausas=int(rng.integers(10, 40)), pause_dur_mean=abs(rng.normal(.18, .06))))
        pq = Path(d) / "m.parquet"; pd.DataFrame(rows).to_parquet(pq, index=False)
        dyn = DynamicEmbeddingLoader(FakeExtractor(), cache_dir=str(Path(d) / "c2"), in_memory=True)
        bundle = DS.make_dataloaders(str(pq), batch_size=8, embedding_loader=dyn, num_workers=0)
        cfg = M.ModelConfig(handcrafted_dim=len(bundle.feature_names), fusion="concat", num_classes=2)
        net = M.build_model(cfg)
        batch = next(iter(bundle.loaders["train"]))
        out = net(handcrafted=batch["handcrafted"], ssl_features=batch["ssl_features"], ssl_mask=batch["ssl_mask"])
        print(f"batch ssl_features={tuple(batch['ssl_features'].shape)}  "
              f"mask={tuple(batch['ssl_mask'].shape)}  logits={tuple(out['logits'].shape)}")
        print("integração dinâmica -> dataset -> model: OK")