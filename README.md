# News-Driven Stock Direction Classifier

Predicts next-day, 5-day, and 20-day stock price direction (up / down) from financial news headlines using a **BERT → 1D CNN → ML Ensemble / Fine-tuned Head** pipeline. Compares frozen vs. fine-tuned BERT and general vs. financial domain models across three return horizons.

## Architecture

```
News headline (text)
       |
       v
[ NLP: BERT Tokenizer ]
  WordPiece tokenization -> token IDs + attention masks
       |
       v
[ NLP + Deep Learning: BERT Encoder (Transfer Learning) ]
  Frozen path  : all 12 layers fixed — fast feature extractor
  Fine-tuned   : layers 0-9 frozen, layers 10-11 trainable
  Output       : (batch, 64 tokens, 768 dims)
       |
       v
[ Deep Learning: 1D CNN ]
  kernel=2 -> bigrams  ("beats estimates")
  kernel=3 -> trigrams ("raises full year")
  kernel=4 -> 4-grams  ("miss on earnings")
  Global Max Pool + concat -> 768-dim feature vector
       |
       v
[ Machine Learning: Prediction Head ]
  Frozen path  -> Soft Voting Ensemble
                  (LR + LightGBM + CatBoost)
  Fine-tuned   -> Linear layer + BCEWithLogitsLoss
       |
       v
    UP (1) / DOWN (0)
```

## Technology Stack

| Layer | Algorithm | Library | Role |
|---|---|---|---|
| **NLP** | BERT tokenizer (WordPiece) | HuggingFace Transformers | Converts headline text → token IDs + attention masks |
| **NLP** | BERT / FinBERT encoder | HuggingFace Transformers | Contextual token embeddings (768-dim) |
| **Deep Learning** | 1D Convolutional Neural Network | PyTorch `nn.Conv1d` | Extracts bigram/trigram/4-gram phrase features |
| **Deep Learning** | Global Max Pooling | PyTorch | Collapses sequence → fixed 768-dim vector |
| **Deep Learning** | Linear head + BCEWithLogitsLoss | PyTorch | End-to-end fine-tuned binary classifier |
| **Machine Learning** | Logistic Regression | scikit-learn | Linear baseline on CNN features |
| **Machine Learning** | LightGBM | LightGBM | Gradient boosted trees |
| **Machine Learning** | CatBoost | CatBoost | Ordered gradient boosting |
| **Machine Learning** | Soft Voting Ensemble | numpy | Average probabilities across all 3 models |
| **Tuning** | Optuna TPE | Optuna | Bayesian hyperparameter search for LightGBM |

## Experiments

| # | Model | NLP | Deep Learning | Machine Learning | BERT Layers | Targets |
|---|---|---|---|---|---|---|
| 1 | Frozen BERT-base + Ensemble | BERT tokenizer, WordPiece embeddings | 1D CNN (bigram/trigram/4-gram) | LR + LightGBM + CatBoost (soft vote) | All 12 frozen | 1d |
| 2 | BERT-base fine-tuned | BERT tokenizer, WordPiece embeddings | BERT encoder + 1D CNN | Linear head + BCEWithLogitsLoss | Last 2 unfrozen | 1d, 5d, 20d |
| 3 | FinBERT fine-tuned | FinBERT tokenizer, financial embeddings | FinBERT encoder + 1D CNN | Linear head + BCEWithLogitsLoss | Last 2 unfrozen | 1d, 5d, 20d |

## Skills Demonstrated

| JD Requirement | Implementation |
|---|---|
| BERT + Transformers | `bert-base-uncased` and `ProsusAI/finbert` via HuggingFace |
| Transfer Learning | Pretrained weights loaded; frozen vs. partial fine-tuning compared |
| Deep Learning (PyTorch) | `BertCNN`, `BertCNNClassifier`, `Dataset`, `DataLoader`, training loop |
| Fine-tuning BERT | Last 2 transformer layers unfrozen, layer-wise LR decay |
| 1D CNN | `nn.Conv1d` with kernel sizes [2,3,4] + Global Max Pool |
| Text Preprocessing | `BertTokenizer` / `AutoTokenizer`, padding, truncation, attention masks |
| Word Embeddings | BERT token embeddings (768-dim) as CNN input channels |
| Scikit-learn | `LogisticRegression`, `StandardScaler`, `classification_report` |
| LightGBM | Gradient boosted classifier on frozen-BERT CNN features |
| CatBoost | Ordered gradient boosting classifier |
| Soft Voting Ensemble | Averaged probabilities across 3 ML models (LR + LightGBM + CatBoost) |
| NumPy / Pandas | Data loading, EDA, feature matrix ops, results aggregation |
| Hyperparameter Tuning | Optuna TPE Bayesian search over LightGBM params (30 trials) |
| Model Comparison | 9+ model configs × 3 targets in single results table + bar chart |
| GPU Optimization | AMP (FP16) with `GradScaler`, `pin_memory`, `non_blocking`, batch size tuning |

## Data

**Source**: `news_contrarian.features` (PostgreSQL / TimescaleDB)

| Column | Description |
|---|---|
| `title` | News headline — NLP input |
| `article_date` | Publication date — used for temporal train/test split |
| `symbol` | Stock symbol |
| `fwd_ret_1d` | Next-day return — noisiest target |
| `fwd_ret_5d` | 5-day return — medium horizon |
| `fwd_ret_20d` | 20-day return — cleanest post-news drift signal |

**Label**: `fwd_ret_Nd > 0` → 1 (up), 0 (down)

**Size**: ~1.4M rows, 6,673 symbols, 2021–2026



## Usage

Open and run `news_stock_predictor.ipynb` top to bottom.

**Notebook sections:**

| # | Section | Description |
|---|---|---|
| 1 | Data Loading | Loads all 3 return targets in one query; strips timezone from `timestamptz` column |
| 2 | EDA | Class balance, headline lengths, articles per year |
| 3 | Model Architecture | `ArticleDataset`, `ArticleDatasetLabelled`, `BertCNN`, `BertCNNClassifier` |
| 4 | Feature Extraction | Frozen BERT+CNN on all headlines (AMP + large batch; cached to disk after first run) |
| 5 | Train/Test Split | Temporal split — train before 2024, test from 2024 |
| 6 | ML Ensemble | LR + LightGBM + CatBoost + soft voting on cached CNN features |
| 7 | Evaluation | Confusion matrices, classification reports, AUC-ROC |
| 8 | Optuna Tuning | 30-trial Bayesian search for LightGBM hyperparameters |
| 9 | Optuna Visualization | AUC over trials, hyperparameter importance chart |
| 10 | Baseline Results | Frozen BERT ensemble comparison table + ROC curves |
| 11 | Fine-Tuning Functions | `train_epoch()` with AMP + gradient clipping, `eval_model()` |
| 12 | Fine-Tuning Comparison | BERT-base + FinBERT × 1d/5d/20d = 6 runs with layer-wise LR decay |
| 13 | Full Results Table | All models × all targets, sorted 1d→5d→20d, best highlighted green |

## ML Ensemble (Soft Voting)

Three models trained on the 768-dim CNN feature vectors, combined by averaging probabilities:

| Model | Type | Key Strength |
|---|---|---|
| Logistic Regression | Linear | Fast, interpretable, good baseline |
| LightGBM | Gradient boosting | Non-linear, fast on high-dimensional features |
| CatBoost | Ordered boosting | Robust to noisy, high-cardinality data |
| **Soft Vote** | Probability average | Diverse inductive biases reduce variance |

## Fine-Tuning Details

**Why unfreeze only the last 2 layers?**
Early BERT layers learn general syntax — shared across all tasks, no benefit to updating them.
Later layers learn task-specific semantics — these adapt to financial headline patterns.
Unfreezing all 110M parameters would overfit on noisy stock direction labels.

**Layer-wise learning rate decay:**
```
BERT layers 10-11 (unfrozen) : lr = 2e-5   (small — protect pretrained weights)
CNN + classifier head        : lr = 2e-4   (10x larger — training new layers from scratch)
```

**Training stabilizers:**
- Gradient clipping (`max_norm=1.0`) — prevents exploding gradients on noisy labels
- `BCEWithLogitsLoss` — numerically stable binary cross-entropy (applies sigmoid internally)
- `weight_decay=0.01` — L2 regularisation via AdamW

## GPU Optimization (RTX 4090)

| Technique | Config | Effect |
|---|---|---|
| Automatic Mixed Precision (AMP) | `USE_AMP=True` | FP16 activations — ~2× throughput, ~2× less VRAM |
| `GradScaler` | Shared across runs | Prevents FP16 gradient underflow during backward pass |
| Large extraction batch | `EXTRACT_BATCH=512` | Fills more VRAM per step (no gradients stored) |
| Large fine-tune batch | `FINETUNE_BATCH=128` | More GPU work per step vs default 64 |
| `pin_memory=True` | All DataLoaders | Pinned host memory — faster CPU→GPU DMA transfer |
| `non_blocking=True` | `.to(DEVICE)` calls | Overlaps memory transfer with GPU compute |

## Feature Caching

Frozen BERT+CNN extraction runs once and is cached to avoid re-running on every experiment:

```
features.npy   — CNN feature matrix  (n_articles x 768)
labels.npy     — 1d binary labels    (n_articles,)
```

Delete these files to force re-extraction (required after changing `MAX_LEN` or `NUM_FILTERS`).

## Configuration

All parameters in the **Config** cell:

| Parameter | Default | Description |
|---|---|---|
| `BERT_MODELS` | `bert-base-uncased`, `ProsusAI/finbert` | Models to compare |
| `TARGETS` | `fwd_ret_1d/5d/20d` | Return horizons to compare |
| `MAX_LEN` | `64` | Max tokens per headline |
| `NUM_FILTERS` | `256` | CNN filters per kernel size |
| `KERNEL_SIZES` | `[2, 3, 4]` | CNN n-gram window sizes |
| `EXTRACT_BATCH` | `512` | Batch size for frozen-BERT feature extraction |
| `FINETUNE_BATCH` | `128` | Batch size for fine-tuning |
| `USE_AMP` | `True` | Enable FP16 mixed precision (requires CUDA) |
| `UNFREEZE_LAST_N` | `2` | BERT layers to unfreeze for fine-tuning |
| `FINETUNE_LR` | `2e-5` | Learning rate for unfrozen BERT layers |
| `FINETUNE_EPOCHS` | `2` | Fine-tuning epochs (2 is standard for BERT) |
| `MAX_TRAIN_SAMPLES` | `200,000` | Max training rows per fine-tuning run |
| `SPLIT_DATE` | `2024-01-01` | Temporal train/test cutoff |
| `N_TRIALS` | `30` | Optuna hyperparameter search trials |

## Requirements

- Python 3.8+
- CUDA GPU recommended (tested on RTX 4090)
- PostgreSQL with `news_contrarian.features` table
- `sqlalchemy` for pandas DB connection
