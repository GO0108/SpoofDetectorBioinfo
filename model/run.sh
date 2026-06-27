#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Pipeline completo de treino com ablação de features
#
# FLUXO:
#   1. export_parquet.py  → exporta o parquet com todas as colunas (incluindo pausas)
#   2. extract_xlsr.py    → extrai embeddings XLS-R e salva em .npy por utterance
#   3. Train.py           → ablação: testa cada subconjunto de features handcrafted
#
#      A ETAPA 3 roda Train.py ~15 vezes sobre o MESMO parquet (uma vez por
#      combinação fusion/feature-subset). Para não repetir a extração+split
#      train/dev/test em cada uma dessas chamadas, todas usam
#      --feature-cache-dir "$OUT/feature_cache": a 1ª chamada extrai e salva
#          $OUT/feature_cache/train_features.parquet
#          $OUT/feature_cache/dev_features.parquet
#          $OUT/feature_cache/test_features.parquet
#      e as chamadas seguintes apenas checam que esses 3 arquivos existem e
#      pulam a extração (ver dataset.py / --feature-cache-dir em Train.py).
#
# USO:
#   bash run_experiments.sh \
#       --metrics  /caminho/metrics_brpseechdf.parquet \
#       --reaper   /caminho/metrics_brpseechdf_reaper.parquet \
#       --wavs     /caminho/para/os/wavs \
#       --out      ./resultados
#
# OPÇÕES:
#   --metrics   Parquet com métricas escalares (extract_metrics.py)
#   --reaper    Parquet com séries REAPER (f0_reaper, corr, times_reaper)
#   --wavs      Diretório raiz dos áudios (para extração XLS-R)
#   --out       Diretório de saída (logs, embeddings, checkpoints)
#   --epochs    Épocas por run (default: 30)
#   --batch     Batch size (default: 32)
#   --workers   Num workers DataLoader (default: 0; aumente após aquecer cache)
#   --device    cpu | cuda | cuda:0 (default: detecta automaticamente)
#   --no-xlsr   Pula extração XLS-R (roda só ablação handcrafted_only)
#   --skip-export  Pula etapa 1 (parquet já exportado em --out/full_metrics.parquet)
#   --skip-xlsr    Pula etapa 2 (embeddings já em --out/xlsr_embeddings/)
#   --no-feature-cache  Desliga o cache train/dev/test da etapa 3 (força
#                       extração em TODA chamada de Train.py; só use se mudar
#                       --metrics/--seed entre runs e quiser garantir um split novo)
#
# CORPUS JÁ DIVIDIDO EM PASTAS train/dev/test (split oficial, sem locutor
# repetido entre elas)? Não use esse fluxo de --metrics/--reaper único (ele
# concatena tudo e REDIVIDE por locutor, perdendo o split oficial). Em vez
# disso, rode f0_extraction.py em CADA pasta separadamente e processe cada
# saída com build_features.py direto para os nomes que o cache espera:
#
#   python3 f0_extraction.py /dados/train --output raw_train.parquet
#   python3 f0_extraction.py /dados/dev   --output raw_dev.parquet
#   python3 f0_extraction.py /dados/test  --output raw_test.parquet
#
#   python3 build_features.py raw_train.parquet --out resultados/feature_cache/train_features.parquet
#   python3 build_features.py raw_dev.parquet   --out resultados/feature_cache/dev_features.parquet
#   python3 build_features.py raw_test.parquet  --out resultados/feature_cache/test_features.parquet
#
#   bash run_experiments.sh --out resultados --skip-export --no-xlsr
#
# (com os 3 arquivos já em resultados/feature_cache/, a ETAPA 3 detecta o
# cache e usa o split oficial direto, sem reextrair nem redividir por locutor)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
METRICS=""
REAPER=""
WAVS=""
OUT="./resultados"
EPOCHS=30
BATCH=32
WORKERS=0
DEVICE=""          # vazio = Train.py detecta sozinho
NO_XLSR=0
SKIP_EXPORT=0
SKIP_XLSR=0
NO_FEATURE_CACHE=0

# --------------------------------------------------------------------------- #
# Parse de argumentos
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --metrics)    METRICS="$2";    shift 2 ;;
        --reaper)     REAPER="$2";     shift 2 ;;
        --wavs)       WAVS="$2";       shift 2 ;;
        --out)        OUT="$2";        shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch)      BATCH="$2";      shift 2 ;;
        --workers)    WORKERS="$2";    shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --no-xlsr)    NO_XLSR=1;       shift 1 ;;
        --skip-export) SKIP_EXPORT=1;  shift 1 ;;
        --skip-xlsr)   SKIP_XLSR=1;    shift 1 ;;
        --no-feature-cache) NO_FEATURE_CACHE=1; shift 1 ;;
        *) echo "[ERRO] Argumento desconhecido: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --------------------------------------------------------------------------- #
# Validações
# --------------------------------------------------------------------------- #
# --metrics só é obrigatório se a ETAPA 1 for rodar de fato (sem --skip-export).
# Com --skip-export + --feature-cache-dir já populado (ex.: split oficial
# pré-extraído com build_features.py), nem $METRICS nem $PARQUET_OUT chegam a
# ser lidos pelo Train.py — então não faz sentido exigir --metrics aqui.
if [[ $SKIP_EXPORT -eq 0 && -z "$METRICS" ]]; then
    echo "[ERRO] --metrics é obrigatório (ou use --skip-export se o parquet"
    echo "       unificado/cache de features já existir)."
    echo "Uso: bash run_experiments.sh --metrics <parquet> --reaper <parquet> --wavs <dir>"
    exit 1
fi
# --reaper só é obrigatório se --metrics NÃO trouxer as colunas REAPER junto
# (times_reaper/f0_reaper/corr). O extract_metrics.py atual já produz tudo num
# parquet só, então nesse caso --reaper pode ficar de fora — a checagem real é
# feita dentro do bloco Python da ETAPA 1 (abaixo), aqui só avisamos.
if [[ $SKIP_EXPORT -eq 0 && -z "$REAPER" ]]; then
    log "  [AVISO] --reaper não informado: assumindo que '$METRICS' já contém as " \
        "colunas REAPER (times_reaper/f0_reaper/corr), como o extract_metrics.py produz."
fi
if [[ $NO_XLSR -eq 0 && $SKIP_XLSR -eq 0 && -z "$WAVS" ]]; then
    echo "[ERRO] --wavs é obrigatório para extrair embeddings XLS-R."
    echo "       Use --no-xlsr para rodar só ablação handcrafted, ou"
    echo "       --skip-xlsr se os .npy já existirem."
    exit 1
fi

# --------------------------------------------------------------------------- #
# Diretórios de saída
# --------------------------------------------------------------------------- #
PARQUET_OUT="$OUT/full_metrics.parquet"
EMB_DIR="$OUT/xlsr_embeddings"
CKPT_DIR="$OUT/checkpoints"
LOG_DIR="$OUT/logs"
CACHE_DIR="$OUT/feature_cache"   # cache do split train/dev/test (ver ETAPA 3)

mkdir -p "$OUT" "$EMB_DIR" "$CKPT_DIR" "$LOG_DIR" "$CACHE_DIR"

# Helper: adiciona --device só se foi passado explicitamente
device_flag() {
    # Train.py detecta cuda/cpu automaticamente; só passamos se o usuário forçou
    if [[ -n "$DEVICE" ]]; then
        echo "--device $DEVICE"
    fi
}

# --------------------------------------------------------------------------- #
# ETAPA 1 — Exportar parquet unificado com colunas de pausa
# --------------------------------------------------------------------------- #
if [[ $SKIP_EXPORT -eq 0 ]]; then
    log "=== ETAPA 1: Exportando parquet com features de pausa ==="
    python3 - <<PYEOF
import ast, sys
import numpy as np
import pandas as pd

metrics_path = "$METRICS"
reaper_path  = "$REAPER"   # pode vir vazio
out_path     = "$PARQUET_OUT"

REAPER_COLS = ["times_reaper", "f0_reaper", "corr"]

def to_array(s):
    if isinstance(s, (list, np.ndarray)):
        return np.asarray(s, dtype=float)
    return np.asarray(ast.literal_eval(s), dtype=float)

print(f"  Carregando métricas: {metrics_path}")
df = pd.read_parquet(metrics_path)

if reaper_path:
    print(f"  Carregando REAPER:    {reaper_path}")
    rea = pd.read_parquet(reaper_path)
    for col in REAPER_COLS:
        if col in rea.columns:
            rea[col] = rea[col].apply(to_array)
    print("  Fazendo merge por 'file'...")
    rea_cols = ["file"] + [c for c in REAPER_COLS if c in rea.columns]
    data = df.merge(rea[rea_cols], on="file", how="inner")
    print(f"  Merge: {len(data)} linhas")
elif all(c in df.columns for c in REAPER_COLS):
    # extract_metrics.py já entrega escalares + REAPER no mesmo parquet —
    # sem merge nenhum, só garante que as colunas de array estão como array.
    print("  --reaper não informado; usando '$METRICS' direto "
          "(já contém times_reaper/f0_reaper/corr).")
    data = df.copy()
    for col in REAPER_COLS:
        data[col] = data[col].apply(to_array)
    print(f"  Linhas: {len(data)}")
else:
    faltam = [c for c in REAPER_COLS if c not in df.columns]
    print(f"[ERRO] --reaper não foi informado e faltam colunas REAPER em "
          f"'$METRICS': {faltam}. Informe --reaper ou gere um parquet que já "
          f"contenha essas colunas (ex.: extract_metrics.py).", file=sys.stderr)
    sys.exit(1)

# ---- Extração das features de pausa (mesma lógica do notebook seção 4) ----
# Se o f0_extraction.py já calculou isso (versão atualizada, que extrai pausa
# no mesmo passe da extração de áudio), as colunas já estão em `data` — não
# recalcula (evitaria duplicar/colidir colunas e reabrir os áudios de novo).
PAUSE_COLS = ["n_pausas", "rms_mean", "rms_std", "zcr_mean", "zcr_std",
              "pause_dur_mean", "pause_dur_std"]

if all(c in data.columns for c in PAUSE_COLS):
    n_ok = data["rms_mean"].notna().sum()
    print(f"  Features de pausa já vêm de '$METRICS' (f0_extraction.py) — "
          f"pulando recálculo. {n_ok}/{len(data)} com sucesso.")
else:
    try:
        import librosa
        from tqdm.auto import tqdm
        HAS_LIBROSA = True
    except ImportError:
        HAS_LIBROSA = False
        print("  [AVISO] librosa não encontrado — features de pausa serão NaN.")
        print("          Instale com: pip install librosa tqdm")

    def extract_pause_features(row):
        """
        f0 < 0 no REAPER = frame unvoiced (inclui consoantes surdas além de silêncio).
        rms_mean das pausas é a feature mais discriminativa encontrada na análise.
        """
        try:
            times = row["times_reaper"]
            f0    = row["f0_reaper"]
            y, sr = librosa.load(row["path"], sr=None)

            pause_mask = f0 < 0
            frame_dur  = np.diff(times).mean() if len(times) > 1 else 0.01
            rms_pausas, zcr_pausas, dur_pausas = [], [], []

            def add_segment(t_start, t_end):
                s, e = int(t_start * sr), int(t_end * sr)
                seg = y[s:e]
                if len(seg) > 0:
                    rms_pausas.append(float(np.sqrt(np.mean(seg ** 2))))
                    zcr_pausas.append(float(np.mean(librosa.feature.zero_crossing_rate(seg)[0])))
                    dur_pausas.append(float(t_end - t_start))

            in_pause, t_start = False, None
            for i, is_pause in enumerate(pause_mask):
                if is_pause and not in_pause:
                    t_start, in_pause = times[i], True
                elif not is_pause and in_pause:
                    add_segment(t_start, times[i - 1] + frame_dur)
                    in_pause = False
            if in_pause:
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
            return {c: np.nan for c in PAUSE_COLS}

    if HAS_LIBROSA:
        print("  Extraindo features de pausa...")
        pause_rows = [extract_pause_features(row) for _, row in tqdm(data.iterrows(), total=len(data))]
        data = pd.concat([data.reset_index(drop=True), pd.DataFrame(pause_rows)], axis=1)
        n_ok = data["rms_mean"].notna().sum()
        print(f"  Pausas extraídas: {n_ok}/{len(data)} com sucesso")
    else:
        for col in PAUSE_COLS:
            data[col] = np.nan

# Agrega o contorno de F0 (REAPER) em f0_mean/f0_std ANTES de descartar
# "f0_reaper" abaixo — mesma lógica de dataset.py:aggregate_f0_row (f0 > 0 =
# vozeado; REAPER marca não-vozeado como -1). Sem isso, dataset.py não tem
# como recalcular f0_mean/f0_std depois (o contorno já não estaria mais aqui).
def _aggregate_f0(f0_reaper):
    f0 = np.asarray(f0_reaper, dtype=float)
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return np.nan, np.nan
    return float(voiced.mean()), float(voiced.std())

print("  Agregando F0 (f0_mean/f0_std a partir do contorno REAPER)...")
f0_aggs = [_aggregate_f0(f0) for f0 in data["f0_reaper"]]
data["f0_mean"] = [a[0] for a in f0_aggs]
data["f0_std"]  = [a[1] for a in f0_aggs]
n_ok_f0 = data["f0_mean"].notna().sum()
print(f"  F0 agregado: {n_ok_f0}/{len(data)} com frames vozeados")

# Remove colunas de array antes de salvar (não serializáveis como parquet simples)
drop_cols = [c for c in ["times_reaper", "f0_reaper", "corr"] if c in data.columns]
data_save = data.drop(columns=drop_cols)
data_save.to_parquet(out_path, index=False)
print(f"  Salvo: {out_path}  ({len(data_save)} linhas, {len(data_save.columns)} colunas)")
print(f"  Colunas: {sorted(data_save.columns.tolist())}")
PYEOF
    log "Etapa 1 concluída → $PARQUET_OUT"
else
    log "Etapa 1 pulada (--skip-export). Usando: $PARQUET_OUT"
fi

# --------------------------------------------------------------------------- #
# ETAPA 2 — Extrair embeddings XLS-R → .npy por utterance
# --------------------------------------------------------------------------- #
if [[ $NO_XLSR -eq 0 && $SKIP_XLSR -eq 0 ]]; then
    log "=== ETAPA 2: Extraindo embeddings XLS-R ==="
    log "    Saída: $EMB_DIR"
    log "    Isso pode demorar (um forward pass por áudio). Use screen/tmux."
    python3 - <<PYEOF
import numpy as np
import pandas as pd
from pathlib import Path

parquet = "$PARQUET_OUT"
wav_dir = "$WAVS"
emb_dir = Path("$EMB_DIR")
emb_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(parquet)
print(f"  {len(df)} utterances a processar")

# Usa o DynamicEmbeddingLoader do Embeddings.py com cache em disco
import sys
sys.path.insert(0, ".")
from Embeddings import XLSRExtractor, DynamicEmbeddingLoader, warm_cache

extractor = XLSRExtractor(
    model_name="facebook/wav2vec2-xls-r-300m",
    layers="last",   # (T, D) — para ssl_features no Dataset.py
    freeze=True,
)
print(f"  Backbone: wav2vec2-xls-r-300m  dim={extractor.dim}")

loader = DynamicEmbeddingLoader(
    extractor,
    path_col="path",
    key_col="file",
    cache_dir=str(emb_dir),
    in_memory=False,   # disco apenas — sem limite de RAM
)

print("  Aquecendo cache (salva .npy por utterance)...")
warm_cache(loader, df, verbose=True)

n_saved = len(list(emb_dir.glob("*.npy")))
print(f"  Embeddings salvos: {n_saved}/{len(df)} em {emb_dir}")
PYEOF
    log "Etapa 2 concluída → $EMB_DIR"
elif [[ $SKIP_XLSR -eq 1 ]]; then
    log "Etapa 2 pulada (--skip-xlsr). Usando embeddings em: $EMB_DIR"
else
    log "Etapa 2 pulada (--no-xlsr). Ablação será só handcrafted_only."
fi

# --------------------------------------------------------------------------- #
# ETAPA 3 — Ablação de features: treino com diferentes subconjuntos
# --------------------------------------------------------------------------- #
log "=== ETAPA 3: Ablação de features ==="

# Cada entrada: "NOME_DO_EXPERIMENTO|--args-extras-para-Dataset"
# O Dataset.py filtra as features via FEATURE_SUBSET (ver abaixo).
# fusion é sempre "concat" quando há XLS-R, "handcrafted_only" sem ele.

HAS_EMB=0
if [[ $NO_XLSR -eq 0 ]]; then
    if ls "$EMB_DIR"/*.npy 1>/dev/null 2>&1; then
        HAS_EMB=1
    fi
fi

# Cache do split train/dev/test (build_feature_frame + grouped_split), reusado
# por TODAS as ~15 chamadas de Train.py abaixo (1ª chamada extrai e salva em
# $CACHE_DIR; as seguintes só checam que os 3 parquets existem e pulam a
# extração). Desligue com --no-feature-cache se mudar --metrics/seed entre runs.
if [[ $NO_FEATURE_CACHE -eq 0 ]]; then
    CACHE_FLAG="--feature-cache-dir $CACHE_DIR"
    log "  Cache de features (train/dev/test): $CACHE_DIR"
else
    CACHE_FLAG=""
    log "  [AVISO] --no-feature-cache: extração será refeita em CADA run."
fi

run_experiment() {
    local NAME="$1"
    local FUSION="$2"
    local SUBSET_FLAG="$3"   # ex: "--feature-subset scalar" (veja patch abaixo)
    local EMB_FLAG="$4"      # "--emb-dir $EMB_DIR" ou vazio

    local LOG="$LOG_DIR/${NAME}.log"
    local CKPT="$CKPT_DIR/${NAME}.pt"

    log "  → $NAME  (fusion=$FUSION)"
    python3 Train.py \
        "$PARQUET_OUT" \
        --fusion "$FUSION" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --feature-loss \
        --patience 7 \
        --out "$CKPT" \
        $SUBSET_FLAG \
        $EMB_FLAG \
        $CACHE_FLAG \
        $(device_flag) \
        2>&1 | tee "$LOG" | grep -E "(ep |Melhor|Teste|features=|ERRO|cache)" || true

    # Extrai métricas finais do log e adiciona à tabela
    local TEST_LINE
    TEST_LINE=$(grep "^Teste:" "$LOG" 2>/dev/null | tail -1 || echo "Teste: N/A")
    echo "$NAME | $FUSION | $TEST_LINE" >> "$OUT/resultados_ablacao.txt"
    log "    $TEST_LINE  → $CKPT"
}

# Configura a flag de embedding dependendo do que está disponível
if [[ $HAS_EMB -eq 1 ]]; then
    EMB_FLAG="--emb-dir $EMB_DIR"
    FUSION_FULL="concat"
    FUSION_HAND="handcrafted_only"
else
    EMB_FLAG=""
    FUSION_FULL="handcrafted_only"
    FUSION_HAND="handcrafted_only"
    log "  [AVISO] Sem embeddings XLS-R — todos os runs usarão handcrafted_only."
fi

echo "" > "$OUT/resultados_ablacao.txt"
echo "EXPERIMENTO | FUSION | MÉTRICAS DE TESTE" >> "$OUT/resultados_ablacao.txt"
echo "$(printf '=%.0s' {1..70})" >> "$OUT/resultados_ablacao.txt"

# --------------------------------------------------------------------------- #
# Grupo A — Só handcrafted (sem XLS-R), ablação por subgrupo de feature
# Testa o poder isolado de cada grupo acústico.
# --------------------------------------------------------------------------- #
log "--- Grupo A: handcrafted_only (ablação por subgrupo) ---"

run_experiment "A1_scalar_only"       "handcrafted_only" "--feature-subset scalar"    ""
run_experiment "A2_pause_only"        "handcrafted_only" "--feature-subset pause"     ""
run_experiment "A3_f0_only"           "handcrafted_only" "--feature-subset f0"        ""
run_experiment "A4_scalar_pause"      "handcrafted_only" "--feature-subset scalar,pause" ""
run_experiment "A5_scalar_f0"         "handcrafted_only" "--feature-subset scalar,f0" ""
run_experiment "A6_pause_f0"          "handcrafted_only" "--feature-subset pause,f0"  ""
run_experiment "A7_all_handcrafted"   "handcrafted_only" "--feature-subset all"       ""

# --------------------------------------------------------------------------- #
# Grupo B — Fusão com XLS-R (se disponível), ablação do ramo handcrafted
# Testa o que cada grupo acústico adiciona ao XLS-R.
# --------------------------------------------------------------------------- #
if [[ $HAS_EMB -eq 1 ]]; then
    log "--- Grupo B: concat XLS-R + ablação handcrafted ---"
    run_experiment "B1_xlsr_only"         "wav2vec_only"    "--feature-subset all"       "$EMB_FLAG"
    run_experiment "B2_xlsr_scalar"       "concat"          "--feature-subset scalar"    "$EMB_FLAG"
    run_experiment "B3_xlsr_pause"        "concat"          "--feature-subset pause"     "$EMB_FLAG"
    run_experiment "B4_xlsr_f0"           "concat"          "--feature-subset f0"        "$EMB_FLAG"
    run_experiment "B5_xlsr_scalar_pause" "concat"          "--feature-subset scalar,pause" "$EMB_FLAG"
    run_experiment "B6_xlsr_all"          "concat"          "--feature-subset all"       "$EMB_FLAG"

    log "--- Grupo C: cross_attention vs concat (fusão completa) ---"
    run_experiment "C1_cross_attn_all"    "cross_attention" "--feature-subset all"       "$EMB_FLAG"
fi

# --------------------------------------------------------------------------- #
# Tabela final
# --------------------------------------------------------------------------- #
log "=== CONCLUÍDO ==="
log "Resultados: $OUT/resultados_ablacao.txt"
echo ""
cat "$OUT/resultados_ablacao.txt"