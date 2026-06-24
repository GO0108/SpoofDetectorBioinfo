"""
extract_metrics.py
------------------
Extrai métricas prosódicas/vocais de todos os wavs em um diretório.

Uso:
    python extract_metrics.py <caminho_wavs> [--output metrics.parquet] [--workers N]
"""

import argparse
import os
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import soundfile as sf
import pyreaper
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

warnings.filterwarnings("ignore")


def extract_one(wav_path: Path) -> dict:
    y_int16, sr = sf.read(str(wav_path), dtype="int16", always_2d=False)
    if y_int16.ndim > 1:
        y_int16 = y_int16.mean(axis=1).astype(np.int16)

    # F0 via REAPER
    _, _, times_reaper, f0_reaper, corr = pyreaper.reaper(y_int16, sr)


    # Praat
    y_f64 = y_int16.astype(np.float64) / 32768.0
    snd   = parselmouth.Sound(y_f64, sampling_frequency=sr)

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    pp_args  = (0, 0, 0.0001, 0.02, 1.3)
    shm_args = (0, 0, 0.0001, 0.02, 1.3, 1.6)
    sp = [snd, point_process]

    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    parts   = wav_path.parts
    model   = parts[-3]
    patient = parts[-2]

    return {
        "file":              wav_path.name,
        "path":              str(wav_path),
        "model":             model,
        "patient":           patient,
        "times_reaper":        times_reaper,
        "f0_reaper":         f0_reaper,
        "corr":      corr,
        "jitter_local_pct":  call(point_process, "Get jitter (local)",    *pp_args) * 100,
        "jitter_rap_pct":    call(point_process, "Get jitter (rap)",      *pp_args) * 100,
        "jitter_ppq5_pct":   call(point_process, "Get jitter (ppq5)",     *pp_args) * 100,
        "jitter_ddp_pct":    call(point_process, "Get jitter (ddp)",      *pp_args) * 100,
        "shimmer_local_pct": call(sp, "Get shimmer (local)",              *shm_args) * 100,
        "shimmer_dB":        call(sp, "Get shimmer (local_dB)",           *shm_args),
        "shimmer_apq3_pct":  call(sp, "Get shimmer (apq3)",              *shm_args) * 100,
        "shimmer_apq5_pct":  call(sp, "Get shimmer (apq5)",              *shm_args) * 100,
        "shimmer_dda_pct":   call(sp, "Get shimmer (dda)",               *shm_args) * 100,
        "hnr_mean_dB":       call(harmonicity, "Get mean", 0, 0),
    }


def process_file(wav_path_str: str):
    """Wrapper para rodar em processo separado."""
    try:
        return "ok", extract_one(Path(wav_path_str)), None
    except Exception:
        return "error", None, (wav_path_str, traceback.format_exc())


def main():
    n_cpus = os.cpu_count() or 1

    parser = argparse.ArgumentParser(description="Extrai métricas de wavs")
    parser.add_argument("wav_dir",   type=str)
    parser.add_argument("--output",  type=str, default="metrics.parquet")
    parser.add_argument("--workers", type=int, default=max(1, n_cpus - 1),
                        help=f"Processos paralelos (default: nCPUs-1 = {max(1, n_cpus-1)})")
    args = parser.parse_args()

    wav_dir    = Path(args.wav_dir)
    out_path   = Path(args.output)
    error_path = out_path.with_name(out_path.stem + "_errors.csv")

    wav_files = sorted(wav_dir.rglob("*.wav"))
    if not wav_files:
        print(f"[AVISO] Nenhum .wav encontrado em: {wav_dir}")
        print("[AVISO] Tentando arquivos .flac...")
        wav_files = sorted(wav_dir.rglob("*.flac"))
        if not wav_files:
            print(f"[ERRO] Nenhum arquivo .flac encontrado em: {wav_dir}")
            return

    print(f"[INFO] {len(wav_files):,} arquivos | {args.workers} workers")

    rows, errors = [], 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, str(p)): p for p in wav_files}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="wav", dynamic_ncols=True):
            status, row, err = fut.result()
            if status == "ok":
                rows.append(row)
            else:
                errors += 1
                with open(error_path, "a") as f:
                    f.write(f"{err[0]}\t{err[1].strip()}\n")

    if rows:
        pd.DataFrame(rows).to_parquet(out_path, index=False, engine="pyarrow")
        print(f"[OK] {len(rows):,} linhas salvas → {out_path}")
    if errors:
        print(f"[ERRO] {errors:,} falhas → {error_path}")


if __name__ == "__main__":
    main()