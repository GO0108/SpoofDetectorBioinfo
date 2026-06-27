"""
extract_metrics.py
------------------
Extrai métricas prosódicas/vocais de todos os wavs em um diretório:
escalares Praat (jitter/shimmer/HNR), features de pausa (n_pausas, rms_mean/
std, zcr_mean/std, pause_dur_mean/std) e agregação de F0 (f0_mean/f0_std) —
tudo no mesmo passe por arquivo (reusa o áudio já carregado).

A saída já tem todas as colunas que Dataset.py/Train.py precisam para treinar
direto (incluindo via --feature-cache-dir, apontando --output para
train_features.parquet / dev_features.parquet / test_features.parquet).

Corpora grandes (centenas de milhares de arquivos, várias horas de extração):
  - RETOMÁVEL: se --output já existe (rodada anterior incompleta/interrompida),
    só processa o que falta. Mate o processo a qualquer momento (Ctrl+C, queda
    de conexão, etc.) e rode o MESMO comando de novo pra continuar.
  - CHECKPOINT: salva o progresso em --output a cada --checkpoint-every
    arquivos OK (default 2000), com troca atômica — nunca corrompe o arquivo.
  - --keep-raw-reaper: por padrão o contorno bruto do REAPER (times_reaper/
    f0_reaper/corr) NÃO é salvo — só os agregados (f0_mean/f0_std), que são o
    que o treino usa. Em 443k arquivos isso evita vários GB extras em memória
    e disco. Passe a flag se quiser o contorno bruto pra análise (notebook).
  - --force: ignora qualquer --output existente e reprocessa tudo do zero.

Estrutura de pastas esperada: .../{split}/{classe}/arquivo.{wav,flac}
    ex.: BRSpeech-DF/dev/bonafide/4067_3313_000000-0001.flac
    -> split="dev", model="bonafide", speaker="4067" (1º token do arquivo)

    split   = parts[-3]  -> qual split oficial do corpus (train/dev/test)
    model   = parts[-2]  -> classe real (bonafide / f5tts / xtts / ...)
    speaker = 1º token do nome do arquivo -> locutor de verdade

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
import librosa
import pyreaper
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

warnings.filterwarnings("ignore")


def extract_pause_features(times: np.ndarray, f0: np.ndarray, y: np.ndarray, sr: int) -> dict:
    """Extrai features acústicas dos segmentos de pausa (f0 < 0 no REAPER).

    Mesma lógica da seção 4 do notebook avaliacao_acustica.ipynb — só que
    aqui reusa o áudio (`y`) e o sample rate (`sr`) já carregados em
    extract_one() em vez de recarregar o arquivo com librosa.load.

    Ressalva: f0 < 0 marca frames *unvoiced* no REAPER, não silêncio puro.
    Consoantes surdas (/s/, /f/, /ʃ/, /p/, /t/, /k/) também caem aqui e têm
    energia real — logo "rms_mean das pausas" mistura silêncio + consoantes
    surdas. Interpretar as conclusões de RMS de pausa com esse cuidado.
    """
    try:
        pause_mask = f0 < 0
        frame_dur = np.diff(times).mean() if len(times) > 1 else 0.01

        rms_pausas, zcr_pausas, dur_pausas = [], [], []

        def add_segment(t_start, t_end):
            s, e = int(t_start * sr), int(t_end * sr)
            segment = y[s:e]
            if len(segment) > 0:
                rms_pausas.append(np.sqrt(np.mean(segment ** 2)))
                zcr_pausas.append(np.mean(librosa.feature.zero_crossing_rate(segment)[0]))
                dur_pausas.append(t_end - t_start)

        in_pause, t_start = False, None
        for i, is_pause in enumerate(pause_mask):
            if is_pause and not in_pause:
                t_start, in_pause = times[i], True
            elif not is_pause and in_pause:
                add_segment(t_start, times[i - 1] + frame_dur)
                in_pause = False
        if in_pause:                       # pausa que vai até o fim do áudio
            add_segment(t_start, times[-1] + frame_dur)

        return {
            "n_pausas":       len(dur_pausas),
            "rms_mean":       float(np.mean(rms_pausas)) if rms_pausas else np.nan,
            "rms_std":        float(np.std(rms_pausas))  if rms_pausas else np.nan,
            "zcr_mean":       float(np.mean(zcr_pausas)) if zcr_pausas else np.nan,
            "zcr_std":        float(np.std(zcr_pausas))  if zcr_pausas else np.nan,
            "pause_dur_mean": float(np.mean(dur_pausas)) if dur_pausas else np.nan,
            "pause_dur_std":  float(np.std(dur_pausas))  if dur_pausas else np.nan,
        }
    except Exception:
        cols = ["n_pausas", "rms_mean", "rms_std", "zcr_mean",
                "zcr_std", "pause_dur_mean", "pause_dur_std"]
        return {c: np.nan for c in cols}


def aggregate_f0(f0_reaper: np.ndarray) -> dict:
    """Agrega o contorno de F0 do REAPER em f0_mean/f0_std (f0 > 0 = vozeado;
    REAPER marca não-vozeado como -1). Mesma lógica de dataset.py:aggregate_f0_row."""
    f0 = np.asarray(f0_reaper, dtype=float)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return {"f0_mean": np.nan, "f0_std": np.nan}
    return {"f0_mean": float(voiced.mean()), "f0_std": float(voiced.std())}


def extract_one(wav_path: Path, keep_raw_reaper: bool = False) -> dict:
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

    # Estrutura de pastas do corpus: .../{split}/{classe}/arquivo.flac
    #   parts[-3] = split   (train / dev / test — pasta do split oficial)
    #   parts[-2] = model   (bonafide / f5tts / xtts / ... — classe real)
    # O locutor de verdade NÃO está em pasta nenhuma: é o 1º token do nome
    # do arquivo (ex.: "4067_3313_000000-0001.flac" -> speaker "4067").
    parts   = wav_path.parts
    split   = parts[-3]
    model   = parts[-2]
    speaker = wav_path.name.split("_")[0]

    row = {
        "file":              wav_path.name,
        "path":              str(wav_path),
        "split":             split,
        "model":             model,
        "speaker":           speaker,
        "jitter_local_pct":  call(point_process, "Get jitter (local)",    *pp_args) * 100,
        # "jitter_rap_pct":    call(point_process, "Get jitter (rap)",      *pp_args) * 100,
        # "jitter_ppq5_pct":   call(point_process, "Get jitter (ppq5)",     *pp_args) * 100,
        # "jitter_ddp_pct":    call(point_process, "Get jitter (ddp)",      *pp_args) * 100,
        "shimmer_local_pct": call(sp, "Get shimmer (local)",              *shm_args) * 100,
        # "shimmer_dB":        call(sp, "Get shimmer (local_dB)",           *shm_args),
        # "shimmer_apq3_pct":  call(sp, "Get shimmer (apq3)",              *shm_args) * 100,
        # "shimmer_apq5_pct":  call(sp, "Get shimmer (apq5)",              *shm_args) * 100,
        # "shimmer_dda_pct":   call(sp, "Get shimmer (dda)",               *shm_args) * 100,
        "hnr_mean_dB":       call(harmonicity, "Get mean", 0, 0),
    }
    # Features de pausa (RMS/ZCR/duração/contagem dos trechos f0<0) — mesma
    # lógica da seção 4 de avaliacao_acustica.ipynb, calculada aqui pra
    # reusar o áudio (y_f64) e o sample rate já carregados acima.
    row.update(extract_pause_features(times_reaper, f0_reaper, y_f64, sr))
    # Agregação de F0 (f0_mean/f0_std) — mesma lógica de dataset.py:aggregate_f0_row.
    row.update(aggregate_f0(f0_reaper))
    # Contorno bruto do REAPER: opcional (--keep-raw-reaper). Não é usado pelo
    # treino (só os agregados acima) e pesa MUITO em corpora grandes — em
    # 443k arquivos isso pode significar vários GB extras em memória/disco.
    if keep_raw_reaper:
        row["times_reaper"] = times_reaper
        row["f0_reaper"] = f0_reaper
        row["corr"] = corr
    return row


def process_file(wav_path_str: str, keep_raw_reaper: bool = False):
    """Wrapper para rodar em processo separado."""
    try:
        return "ok", extract_one(Path(wav_path_str), keep_raw_reaper), None
    except Exception:
        return "error", None, (wav_path_str, traceback.format_exc())


def main():
    n_cpus = os.cpu_count() or 1

    parser = argparse.ArgumentParser(description="Extrai métricas de wavs")
    parser.add_argument("wav_dir",   type=str)
    parser.add_argument("--output",  type=str, default="metrics.parquet")
    parser.add_argument("--workers", type=int, default=max(1, n_cpus - 1),
                        help=f"Processos paralelos (default: nCPUs-1 = {max(1, n_cpus-1)})")
    parser.add_argument("--force", action="store_true",
                        help="reprocessa TUDO mesmo se --output já existir (default: retoma de onde parou)")
    parser.add_argument("--keep-raw-reaper", action="store_true",
                        help="mantém times_reaper/f0_reaper/corr no parquet final (não usado pelo treino; "
                             "pesa MUITO em corpora grandes — default: descarta, fica só com os agregados)")
    parser.add_argument("--checkpoint-every", type=int, default=2000,
                        help="salva o progresso em --output a cada N arquivos OK (default: 2000). "
                             "Se o processo cair, retome rodando o mesmo comando de novo.")
    args = parser.parse_args()

    out_path = Path(args.output)

    # Retomada: se --output já existe (de uma rodada anterior/interrompida),
    # carrega o que já foi processado e só extrai o que falta. --force ignora
    # isso e reprocessa tudo do zero.
    rows = []
    done_paths = set()
    if out_path.exists():
        if args.force:
            print(f"[force] ignorando '{out_path}' existente — reprocessando tudo.")
        else:
            existing = pd.read_parquet(out_path)
            rows = existing.to_dict("records")
            done_paths = set(existing["path"].astype(str))
            print(f"[resume] '{out_path}' já tem {len(done_paths):,} arquivo(s) — "
                  f"retomando de onde parou (use --force para refazer tudo).")

    wav_dir    = Path(args.wav_dir)
    error_path = out_path.with_name(out_path.stem + "_errors.csv")

    wav_files = sorted(wav_dir.rglob("*.wav"))
    if not wav_files:
        print(f"[AVISO] Nenhum .wav encontrado em: {wav_dir}")
        print("[AVISO] Tentando arquivos .flac...")
        wav_files = sorted(wav_dir.rglob("*.flac"))
        if not wav_files:
            print(f"[ERRO] Nenhum arquivo .flac encontrado em: {wav_dir}")
            return

    todo = [p for p in wav_files if str(p) not in done_paths]
    print(f"[INFO] {len(wav_files):,} arquivos totais | {len(done_paths):,} já prontos | "
          f"{len(todo):,} a processar | {args.workers} workers")

    if not todo:
        print(f"[OK] nada a fazer — todos os arquivos já estão em {out_path}.")
        return

    def save_checkpoint():
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(str(tmp), str(out_path))   # troca atômica: nunca deixa --output corrompido

    errors = 0
    since_checkpoint = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, str(p), args.keep_raw_reaper): p for p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="wav", dynamic_ncols=True):
            status, row, err = fut.result()
            if status == "ok":
                rows.append(row)
                since_checkpoint += 1
            else:
                errors += 1
                with open(error_path, "a") as f:
                    f.write(f"{err[0]}\t{err[1].strip()}\n")

            if since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                since_checkpoint = 0

    if rows:
        save_checkpoint()
        print(f"[OK] {len(rows):,} linhas no total salvas → {out_path}")
    if errors:
        print(f"[ERRO] {errors:,} falhas → {error_path}")


if __name__ == "__main__":
    main()