# SpoofDetectorBioinfo

Análise acústica de fala sintética (*deepfake audio / spoofing*) a partir do dataset **BRSpeech-DF**, com foco na extração e comparação de parâmetros escalares de qualidade vocal e no estudo do acoplamento F0–energia entre fala real e fala sintetizada por diferentes modelos de síntese de voz (TTS/voice conversion).

O objetivo é investigar se características de baixo nível — jitter, shimmer, HNR, energia (RMS) e contorno de F0, bem como a coerência entre envoltória de energia e F0 — permitem diferenciar fala humana de fala gerada artificialmente, servindo como base para trabalhos futuros de detecção de spoofing (*anti-spoofing*).

## Estrutura do repositório

```
SpoofDetectorBioinfo/
├── avaliacao_acustica.ipynb        # Etapa 1 — parâmetros escalares
├── avaliacao_acomplamento.ipynb    # Etapa 2 — acoplamento F0–energia
├── feature_extraction/             # Pré-processamento e extração de features
└── model/                          # Preparação para classificação (trabalho futuro)
```

### `avaliacao_acustica.ipynb`

Corresponde à **Etapa 1** do fluxo de análise (Seção III-D). Realiza a extração e comparação dos parâmetros escalares de qualidade vocal entre áudios reais e falsos (sintetizados):

- **Jitter**, **shimmer**, **HNR** (*Harmonic-to-Noise Ratio*), **RMS das pausas** e **F0**;
- Tabela de **medianas** (real vs. falso);
- **Gráficos de violino** para visualização das distribuições;
- **Divergência KL (Kullback-Leibler)** entre real e falso, por modelo de síntese;
- **Gráficos de violino por modelo de síntese**.

### `avaliacao_acomplamento.ipynb`

Corresponde à **Etapa 2** do fluxo de análise (Seção III-D). Analisa o **acoplamento entre F0 e energia** do sinal de voz:

- **Envoltória de energia** (RMS por frame);
- **Coerência** entre F0 e energia via `scipy.signal.coherence` (estimador de Welch);
- **Curva média Cxy(f)** e escalares de **coerência média global** e **coerência prosódica**, por modelo de síntese;
- **Autocorrelação normalizada** das envoltórias de RMS e F0 na janela de **0–500 ms**;
- Escalar de **pico silábico** na banda **120–320 ms**.

### `feature_extraction/`

Scripts de pré-processamento e extração de features dos arquivos `.flac` do **BRSpeech-DF**, incluindo:

- Chamada ao **REAPER** para estimativa de F0;
- Extração via **Parselmouth/Praat** dos parâmetros glotais (jitter, shimmer, HNR, entre outros).

### `model/`

Código preparatório para classificação (extração e organização das features geradas nas etapas anteriores). **O treinamento de um classificador está fora do escopo deste trabalho e constitui trabalho futuro.**

## Dataset

As análises utilizam o **BRSpeech-DF**, conjunto de dados de fala em português contendo amostras reais e amostras sintetizadas por diferentes modelos de síntese de voz (TTS/voice conversion), usado como base para avaliação de técnicas de detecção de spoofing de fala.

## Requisitos

- Python 3.x
- Jupyter Notebook
- [REAPER](https://github.com/google/REAPER) (estimativa de F0)
- [Parselmouth](https://github.com/YannickJadoul/Parselmouth) (interface Python para o Praat)
- Bibliotecas científicas: `numpy`, `scipy`, `pandas`, `matplotlib`/`seaborn`

## Como usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/GO0108/SpoofDetectorBioinfo.git
   cd SpoofDetectorBioinfo
   ```
2. Instale as dependências necessárias (REAPER, Parselmouth e bibliotecas Python).
3. Execute os scripts em `feature_extraction/` para gerar as features a partir dos arquivos `.flac` do BRSpeech-DF.
4. Rode os notebooks na ordem do fluxo de análise:
   - `avaliacao_acustica.ipynb` (Etapa 1)
   - `avaliacao_acomplamento.ipynb` (Etapa 2)
5. Os artefatos gerados em `model/` podem servir de ponto de partida para treinar um classificador de spoofing em trabalhos futuros.

## Trabalho futuro

- Treinamento e avaliação de um classificador de spoofing utilizando as features extraídas em `model/`.
- Extensão da análise a outros modelos de síntese e a outros idiomas/datasets.

## Bônus: experimentos de classificação (ablação de features)

Além das duas etapas de análise exploratória, foram executados experimentos preliminares de classificação (real vs. falso) utilizando diferentes combinações de grupos de features extraídas nas etapas anteriores (features escalares/glotais, features de pausa e features de F0). Cada experimento corresponde a um script de treinamento que salva um checkpoint em `../results/checkpoints/` e um log com as métricas de validação e teste.

### Resultados (validação)

| Experimento | Features utilizadas | EER (val) | AUC (val) |
|---|---|---|---|
| A1_scalar_only | Somente features escalares (jitter, shimmer, HNR, etc.) | 0.2703 | 0.7776 |
| A2_pause_only | Somente features de pausa | — * | — * |
| A3_f0_only | Somente features de F0 | 0.4022 | 0.6472 |
| A4_scalar_pause | Escalares + pausa | 0.1753 | 0.9085 |
| A5_scalar_f0 | Escalares + F0 | 0.2384 | 0.8295 |
| A6_pause_f0 | Pausa + F0 | 0.2402 | 0.8296 |
| A7_all_handcrafted | Todas as features artesanais combinadas | **0.1555** | **0.9246** |

\* O log `A2_pause_only.log` não registrou as métricas finais (apenas um *warning* do PyTorch), sendo necessário reexecutar o experimento para obter os resultados.

Para referência, os resultados no conjunto de **teste** do melhor experimento (A7_all_handcrafted) foram EER=0.1527, AUC=0.9313, acc=0.852, f1=0.779 — indicando que a combinação de todas as features artesanais supera qualquer subconjunto isolado, e que features de pausa são as que mais contribuem isoladamente (A4 supera A5 e A6).

### Descrição do experimento

Cada experimento treina um classificador simples (real vs. falso) usando um subconjunto específico de features artesanais extraídas em `feature_extraction/` e organizadas em `model/`, permitindo comparar a contribuição de cada grupo de features (escalares, pausas, F0) isoladamente e combinado. As métricas reportadas são:

- **EER** (*Equal Error Rate*): ponto em que a taxa de falsos positivos iguala a taxa de falsos negativos — quanto menor, melhor;
- **AUC** (*Area Under the ROC Curve*): quanto maior, melhor a separabilidade entre as classes.

### Como rodar

Os experimentos são orquestrados pelo script `model/run_experiments.sh`, que executa o pipeline completo em 3 etapas:

1. **`export_parquet.py`** — exporta um parquet unificado com todas as colunas (escalares + pausas + F0/REAPER);
2. **`extract_xlsr.py`** — extrai embeddings XLS-R e salva um `.npy` por utterance (opcional, ver `--no-xlsr`);
3. **`Train.py`** — roda a ablação propriamente dita, treinando um classificador para cada subconjunto de features handcrafted (e, se houver embeddings XLS-R, também as fusões dos grupos B/C).

Os experimentos **A1–A7** (cujos resultados estão na tabela acima) correspondem ao **Grupo A** do script: treino *handcrafted_only*, variando o subconjunto de features via `--feature-subset`:

| Experimento | `--feature-subset` |
|---|---|
| A1_scalar_only | `scalar` |
| A2_pause_only | `pause` |
| A3_f0_only | `f0` |
| A4_scalar_pause | `scalar,pause` |
| A5_scalar_f0 | `scalar,f0` |
| A6_pause_f0 | `pause,f0` |
| A7_all_handcrafted | `all` |

O script também define **Grupo B** (fusão `concat` de XLS-R com cada subconjunto handcrafted) e **Grupo C** (`cross_attention` com todas as features), executados automaticamente quando embeddings XLS-R estão disponíveis — mas esses ainda não foram rodados/reportados neste README.

#### Uso básico (pipeline completo, do zero)

```bash
cd model
bash run_experiments.sh \
    --metrics  /caminho/metrics_brpseechdf.parquet \
    --reaper   /caminho/metrics_brpseechdf_reaper.parquet \
    --wavs     /caminho/para/os/wavs \
    --out      ./resultados
```

#### Rodando só a ablação handcrafted (sem extrair XLS-R)

Útil para reproduzir apenas os experimentos A1–A7:

```bash
bash run_experiments.sh \
    --metrics /caminho/metrics_brpseechdf.parquet \
    --out     ./resultados \
    --no-xlsr
```

#### Usando um split oficial (train/dev/test já dividido por pastas)

Se o corpus já vem dividido em pastas `train/dev/test` (sem locutor repetido entre elas), **não** use `--metrics`/`--reaper` únicos — isso concatena tudo e redivide por locutor, perdendo o split oficial. Em vez disso, gere o cache de features diretamente para cada split com `f0_extraction.py` (calcula escalares, pausa e F0 em um único passe) e depois pule as etapas 1 e 2:

```bash
python3 f0_extraction.py /dados/train --output resultados/feature_cache/train_features.parquet
python3 f0_extraction.py /dados/dev   --output resultados/feature_cache/dev_features.parquet
python3 f0_extraction.py /dados/test  --output resultados/feature_cache/test_features.parquet

bash run_experiments.sh --out resultados --skip-export --no-xlsr
```

#### Outras opções relevantes

| Flag | Descrição |
|---|---|
| `--epochs` | Épocas por run (default: 30) |
| `--batch` | Batch size (default: 32) |
| `--workers` | Workers do DataLoader (default: 0) |
| `--device` | `cpu` \| `cuda` \| `cuda:0` (default: detecção automática) |
| `--skip-export` | Pula a etapa 1 (usa `--out/full_metrics.parquet` já existente) |
| `--skip-xlsr` | Pula a etapa 2 (usa embeddings já extraídos em `--out/xlsr_embeddings/`) |
| `--no-feature-cache` | Desliga o cache de train/dev/test da etapa 3 (força reextração em toda chamada de `Train.py`) |

Todas as ~7 (ou ~15, com XLS-R) chamadas de `Train.py` reutilizam o mesmo split de features via `--feature-cache-dir "$OUT/feature_cache"`: a primeira chamada extrai e salva `train_features.parquet`, `dev_features.parquet` e `test_features.parquet`; as chamadas seguintes apenas detectam que os arquivos já existem e pulam a reextração.

Cada execução salva:
- o checkpoint em `resultados/checkpoints/<experimento>.pt`;
- o log completo em `resultados/logs/<experimento>.log`, com a métrica de validação (`Melhor val: EER=... AUC=...`) e de teste (`Teste: EER=... AUC=... acc=... f1=...`);
- curvas de treino/validação/teste no TensorBoard, em `resultados/tensorboard` (visualize com `tensorboard --logdir resultados/tensorboard`);
- uma tabela-resumo consolidada em `resultados/resultados_ablacao.txt`.

