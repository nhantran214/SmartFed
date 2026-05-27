# SmartFed

SmartFed is a federated learning codebase focused on deepfake detection and related benchmarks. Clients use a shared **EfficientNet-B0** image backbone (`torchvision`). It supports **GenImage** (binary real/fake, non-IID across clients) and **CIFAR-10** (10-way classification, IID-style splits), plus optional utilities for other datasets. The main training entry point is `main.py`; a **FedAvg-only** script is `train_fedavg.py`.

This file is the **only** project documentation: setup, data layout, Ollama/LLM features, and testing.

---

## Features

- **Backbone**: **EfficientNet-B0** from `torchvision` for all clients (see `config.py` → `PRIVATE_NETS_NAME_LIST` / `NETS_NAME_LIST`, and `Dataset/utils.py` → `init_nets`). Image inputs are resized to **224×224** for GenImage-style loaders.
- **Hybrid federated training** (`main.py`): private client data, public data where configured, logging, and checkpoints under `Model_Storage/` and `Logs/`.
- **Learnable client weights** (`main.py`): per-client `weight_quality` updated with gradient descent across communication rounds (see [Client aggregation weights](#client-aggregation-weights-weight_quality)).
- **FedAvg** (`train_fedavg.py`): federated averaging using the same `config.py` and dataloaders.
- **LLM-based client selection** (`llm_client_selection.py`): reward model plus strategies (`greedy`, `diversity`, `adaptive`, `ollama`, `llama3`).
- **Loss weighting agent** (`loss_weighting_agent.py`): dynamic per-client loss coefficients via **Ollama** (with rule-based fallback).

---

## Installation

```bash
pip install -r requirements.txt
```

Use a **GPU** when possible. PyTorch and torchvision are required (EfficientNet-B0 uses `torchvision.models.efficientnet_b0`).

---

## Quick start

1. **Configure `config.py`** — follow [Declaring `config.py` for a successful run](#declaring-configpy-for-a-successful-run) so dataset paths, client count, and model settings match your machine and data layout.
2. Prepare data on disk as in [Dataset layout on disk](#dataset-layout-on-disk) (paths must match what you set under `PRIVATE_DATASET_DIR` / `PUBLIC_DATASET_DIR`).
3. Optional pretraining:

   ```bash
   python pretrain.py
   ```

4. Main training:
   The main.py file will be released after the paper is officially accepted.
   
   ```bash
   python main.py
   ```

5. FedAvg-only:

   ```bash
   python train_fedavg.py
   ```

---

## Declaring `config.py` for a successful run

All training scripts (`main.py`, `pretrain.py`, `train_fedavg.py`) read the **`Config`** class in `config.py`. There is no separate YAML or CLI override: **anything you need to change for your environment should be edited in this file.**

### 1. How `config.py` is structured

- **`_SMARTFED_ROOT`**: absolute path to the SmartFed repository (the folder that contains `config.py`). Dataset directories default to **`os.path.join(_SMARTFED_ROOT, 'data', ...)`**.
- **`Config`**: a class whose body runs top to bottom. **`DATASET_TYPE`** selects **one** branch (`'genimage'` or `'cifar10'`). Only the variables in that branch are defined for dataset-specific paths and names.

To use data **outside** the repo, replace the `os.path.join(_SMARTFED_ROOT, ...)` expressions in the active branch with your own absolute paths (or define a variable like `MY_DATA = '/mnt/datasets'` and join from that).

---

### 2. Minimal checklist before the first run

| Step | What to verify |
|------|----------------|
| Dataset mode | Set **`DATASET_TYPE`** to `'genimage'` or `'cifar10'`. |
| Data on disk | Folders exist and match [Dataset layout on disk](#dataset-layout-on-disk) for the chosen mode. |
| Paths | **`PRIVATE_DATASET_DIR`** and **`PUBLIC_DATASET_DIR`** (in the active `if` / `elif` branch) point to those folders. |
| Client count | **`N_CLIENTS`** equals the number of clients you train; for GenImage it must match **`len(DATASET_LIST)`**. |
| Classes vs model | **`OUTPUT_CHANNEL`** and **`PRIVATE_OUTPUT_CHANNEL`** equal `len(PRIVATE_DATASET_CLASSES)` (2 for GenImage, 10 for CIFAR-10 as shipped). |
| Network names | **`PRIVATE_NETS_NAME_LIST`** and **`NETS_NAME_LIST`** each have length **`N_CLIENTS`**. The default setup uses **`EfficientNetB0`** for every client; names are resolved in `Dataset/utils.py` → `init_nets`. |
| `main.py` checkpoints | **`main.py`** expects per-client checkpoints under **`Model_Storage/...`** (see §7). Run **`pretrain.py`** first with matching settings, or place `.ckpt` files manually. **`train_fedavg.py`** starts from freshly initialized nets (no `Model_Storage` load unless you change the code). |

---

### 3. Global training and federated parameters

These apply regardless of dataset (see `config.py` for current numeric defaults).

| Variable | Meaning |
|----------|---------|
| **`SEED`** | Random seed for NumPy / PyTorch (reproducibility). |
| **`N_CLIENTS`** | Number of federated clients; loops in `main.py` / `pretrain.py` / `train_fedavg.py` iterate `0 .. N_CLIENTS-1`. |
| **`TRAIN_BATCH_SIZE`** | Batch size for **public** data loaders (where used with public train/test). |
| **`TEST_BATCH_SIZE`** | Batch size for **public** test paths that use this size. |
| **`TRAIN_BATCH_SIZE_PRIVATE`** | Batch size for **each client’s** private training loader. |
| **`TEST_BATCH_SIZE_PRIVATE`** | Batch size for **each client’s** private test loader. |
| **`COMMUNICATION_EPOCH`** | Number of federated communication rounds (outer loop). |
| **`INTRA_DATASET`** | **`False`**: default split logic (e.g. CIFAR split across clients when not using intra-dataset sampling). **`True`**: uses `PRIVATE_DATA_LEN`-limited index sampling per client for the intra-dataset path in `main.py` / `pretrain.py` / `train_fedavg.py`. Set according to the experiment you are reproducing. |

**GPU memory:** if you run out of VRAM, lower **`TRAIN_BATCH_SIZE_PRIVATE`** and/or **`TRAIN_BATCH_SIZE`** first, then communication batch sizes.

---

### 4. Optimizer, loss, and participant parameters

| Variable | Meaning |
|----------|---------|
| **`PARTICIPANT_PARAMS`** | Dict used everywhere local training is built: **`'loss_function'`** (e.g. `'CE'` or `'SCE'` where implemented), **`'optimizer_name'`** (`'Adam'` or `'SGD'`), **`'learning_rate'`**. Must stay consistent with what you used in **`pretrain.py`** if you load those checkpoints. |
| **`CLIENT_CONFIDENCE_REWEIGHT`** | Enables client confidence reweighting logic tied to **`BETA`** in `main.py`. |
| **`CLIENT_CONFIDENCE_REWEIGHT_LOSS`** | Loss key used in that path (typically `'CE'`). |
| **`BETA`** | Automatically `0.5` when **`CLIENT_CONFIDENCE_REWEIGHT`** is `True`, else `0`. |

---

### 5. Label noise (optional)

| Variable | Meaning |
|----------|---------|
| **`NOISE_TYPE`** | `None`, or a string such as `'pairflip'` / `'symmetric'` if your pipeline injects label noise (must match what `get_dataloader` / dataset code expects). |
| **`NOISE_RATE`** | Noise rate (used with `NOISE_TYPE`). |

These appear in dataloader calls and in **checkpoint directory names** (see §7). With the default **`NOISE_TYPE = None`** and **`NOISE_RATE = 0`**, the folder segment is literally **`None0`**.

---

### 6. Model architecture lists

| Variable | Meaning |
|----------|---------|
| **`PRIVATE_NETS_NAME_LIST`** | Length **`N_CLIENTS`**: per-client architecture for private models and checkpoint filenames. Default: **`['EfficientNetB0'] * N_CLIENTS`**. |
| **`NETS_NAME_LIST`** | Same length; used when building participant nets in pretrain / FedAvg. Keep aligned with **`PRIVATE_NETS_NAME_LIST`**. |

Additional architectures (`ResNet10`, `EfficientNetB4`, etc.) exist in **`Dataset/utils.py`** → **`init_nets`** for experiments, but **SmartFed is intended to run with EfficientNet-B0**.

---

### 7. Checkpoints and `Model_Storage` layout (`pretrain.py` / `main.py`)

**`pretrain.py`** writes:

- Per-client:  
  `Model_Storage/<PARTICIPANT_PARAMS['loss_function']>/<NOISE_TYPE><NOISE_RATE>/<PRIVATE_NETS_NAME_LIST[i]>_<i>.ckpt`
- Optional public model path used in pretrain:  
  `.../public_model.ckpt` under the same `<loss>/<noise>` folder.

**`main.py`** loads each client’s checkpoint from the **same pattern** before continuing training. So for a successful **`main.py`** run you must either:

1. Run **`pretrain.py`** first with the **same** `PARTICIPANT_PARAMS`, `NOISE_TYPE`, `NOISE_RATE`, `PRIVATE_NETS_NAME_LIST`, and `N_CLIENTS`, or  
2. Place compatible `.ckpt` files at those paths manually.

If any path segment does not match, you will get a **file not found** error at startup.

---

### 8. When `DATASET_TYPE == 'genimage'`

Set **`DATASET_TYPE = 'genimage'`** at the top of the dataset block. The following symbols are then active:

| Variable | You must ensure |
|----------|------------------|
| **`DATASET_LIST`** | List of **subdirectory names** under **`PRIVATE_DATASET_DIR`**, one per client, length **`N_CLIENTS`**. Default lists eight generator folders (`adm`, `BigGAN`, …). Each must contain `train/nature`, `train/ai`, `val/nature`, `val/ai`. |
| **`PRIVATE_DATASET_DIR`** | Parent directory of all client folders. Default: `<repo>/data/genimage_medium_fulltest`. |
| **`PRIVATE_DATASET_NAME`** | Must stay **`'genimage'`** for the current `genimage` loader. |
| **`PRIVATE_DATA_LEN`** | Used when **`INTRA_DATASET`** is `True` (cap on indices per client). For `False`, behavior depends on full dataset size. |
| **`PRIVATE_DATASET_CLASSES`** | **`['real', 'fake']`** → binary classification; must match **`OUTPUT_CHANNEL` / `PRIVATE_OUTPUT_CHANNEL`**. |
| **`OUTPUT_CHANNEL`**, **`PRIVATE_OUTPUT_CHANNEL`** | Set to **`len(PRIVATE_DATASET_CLASSES)`** (2). |
| **`PUBLIC_DATASET_NAME`** | **`'genimage'`** for the public loader. |
| **`PUBLIC_DATASET_DIR`** | Default: `<repo>/data/genimage_medium_fulltest/genimage_pub` with the **same** folder layout as a single client root. |
| **`PUBLIC_DATASET_LENGTH`** | Size hint / subsampling for public data where the training code uses it (default `8000`). |

**Consistency:** **`len(DATASET_LIST) == N_CLIENTS`**. If you add a ninth client folder, add its name to **`DATASET_LIST`** and set **`N_CLIENTS = 9`**.

---

### 9. When `DATASET_TYPE == 'cifar10'`

Set **`DATASET_TYPE = 'cifar10'`**.

| Variable | You must ensure |
|----------|------------------|
| **`DATASET_LIST`** | Typically **`['cifar10'] * N_CLIENTS`**: all clients read from the same image root; splits are done by indices in code. |
| **`PRIVATE_DATASET_DIR`** | Default `<repo>/data/cifar` with **`train/<class>/`** and **`test/<class>/`**. Run `Dataset/cifar10_to_images.py` if you only have pickle batches. |
| **`PRIVATE_DATASET_NAME`** | **`'cifar10'`** so `main.py` uses the CIFAR dataloader branch. |
| **`PRIVATE_DATA_LEN`** | Per-client training sample cap when **`INTRA_DATASET`** is `True`; when `False`, CIFAR splitting uses full train set divided by **`N_CLIENTS`**. |
| **`PRIVATE_DATASET_CLASSES`** | The ten CIFAR class folder names; order defines label indices. |
| **`OUTPUT_CHANNEL`**, **`PRIVATE_OUTPUT_CHANNEL`** | **`10`**. |
| **`PUBLIC_DATASET_NAME`** | **`'cifar10'`**. |
| **`PUBLIC_DATASET_DIR`** | Usually the **same** root as private (`<repo>/data/cifar`). |
| **`PUBLIC_DATASET_LENGTH`** | Public subset size where applicable (default `8000`). |

---

### 10. Relationship summary (common mistakes)

1. **`N_CLIENTS`** ≠ **`len(DATASET_LIST)`** (GenImage) → wrong client folders or index errors.  
2. **`OUTPUT_CHANNEL`** ≠ number of classes → shape mismatch in the classifier or loss.  
3. **`PRIVATE_NETS_NAME_LIST`** shorter/longer than **`N_CLIENTS`** → `init_nets` or checkpoint index mismatch.  
4. **`main.py`** without pretrain → missing **`Model_Storage/...`** files.  
5. **`NOISE_TYPE`** / **`NOISE_RATE`** changed after pretrain → checkpoint path no longer matches.  
6. Data not under `<repo>/data/...` → update the **`os.path.join(_SMARTFED_ROOT, ...)`** lines in the active **`DATASET_TYPE`** branch.

After any change, re-run **`pretrain.py`** if you depend on freshly written checkpoints, then **`main.py`** or **`train_fedavg.py`**.

---

## Dataset layout on disk

Training expects **fixed subdirectory names** unless you change `Dataset/init_dataset.py` or `Dataset/cifar10_dataset.py`.

### GenImage (`DATASET_TYPE = 'genimage'`)

Each client is **`PRIVATE_DATASET_DIR / <name>`** where `<name>` matches `Config.DATASET_LIST` (e.g. `adm`, `BigGAN`, `glide`, `midjourney`, `sdv4`, `sdv5`, `vqdm`, `wukong`).

Under each client root:

```text
<client_root>/
├── train/
│   ├── nature/     # real → label 0
│   └── ai/         # fake → label 1
└── val/
    ├── nature/
    └── ai/
```

**Public** data (`PUBLIC_DATASET_DIR`) uses the same layout under one folder (e.g. `.../genimage_pub`).

Runtime path: `os.path.join(PRIVATE_DATASET_DIR, DATASET_LIST[client_index])`.

### CIFAR-10 (`DATASET_TYPE = 'cifar10'`)

Single image root (often shared for private and public), with **one folder per class**:

```text
<cifar_root>/
├── train/
│   ├── airplane/ … truck/
└── test/
    ├── airplane/ … truck/
```

From official CIFAR-10 Python pickles in `data/cifar-10-batches-py/`:

```bash
cd Dataset
python cifar10_to_images.py
```

Defaults: read `../data/cifar-10-batches-py`, write `../data/cifar/`. Align `PRIVATE_DATASET_CLASSES` and `OUTPUT_CHANNEL` with 10 classes (see [Declaring `config.py` for a successful run](#declaring-configpy-for-a-successful-run), §9).

### Celeb-DF v2 (optional loader)

`celebdf2` in `Dataset/init_dataset.py` expects:

```text
<root>/
├── train/real/  train/fake/
└── test/test_real/  test/test_fake/
```

Not enabled in `config.py` by default; wire `dataset='celebdf2'` in `get_dataloader` if you use it.

---

## Dataset comparison

| Aspect | GenImage | CIFAR-10 |
|--------|----------|----------|
| Task | Deepfake binary (real / fake) | 10-class objects |
| Input (in code) | 224×224 after transforms | 32×32 |
| Split style | Different sources per client (non-IID) | Shared pool, index splits (IID-style) |
| Classes | 2 | 10 |

---

## FedAvg training (`train_fedavg.py`)

Uses `config.py` for rounds, batches, models, and paths. Implements local training, weighted **FedAvg** aggregation, periodic evaluation, logs in `Logs/`, and checkpoints in `Model_Storage/`. Helpers include `local_training`, `fedavg_aggregate`, `evaluate_model`, and cross-client evaluation aligned with `Dataset/utils.py` and `loss.py`.

---

## LLM-based client selection (`llm_client_selection.py`)

### Behavior

- **Reward model**: small MLP scoring clients from features (loss, accuracy, data quality, trends, resource/diversity proxies, selection counts, etc.).
- **Strategies**
  - **`greedy`**: highest reward scores.
  - **`diversity`**: balances score and diversity.
  - **`adaptive`**: history-aware exploration/exploitation.
  - **`ollama`** / **`llama3`**: Ollama-backed natural-language selection; falls back to **adaptive** if the API is unavailable.

### Example

```python
from llm_client_selection import create_llm_client_selector

selector = create_llm_client_selector(
    n_clients=8,
    strategy="adaptive",  # or "llama3", "ollama", "greedy", "diversity"
    logger=logger,
)
selected = selector.select_clients(
    current_round=epoch_index,
    selection_budget=8,
    global_performance=0.8,
)
selector.update_client_metrics(
    client_id=0,
    loss=0.5,
    accuracy=0.85,
    data_quality=1.0,
    performance_trend=0.0,
    resource_availability=1.0,
    model_diversity=0.5,
)
if epoch_index > 0:
    selector.train_reward_model(client_metrics, performance_feedback)
```

### Statistics

```python
stats = selector.get_selection_statistics()
# selection_counts, avg_reward_scores, total_rounds, selection_diversity, ...
```

### Tests

```bash
python test_llm_client_selection.py
python test_ollama_integration.py   # requires Ollama (see below)
```

---

## Ollama setup (LLM selection and loss weighting)

Both **client selection** (`ollama` / `llama3`) and **`loss_weighting_agent`** talk to Ollama over HTTP (default `http://localhost:11434`).

### Install Ollama

**Linux / macOS:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:** installer from [https://ollama.ai/download](https://ollama.ai/download).

**Docker:**

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Pull and run a model

```bash
ollama pull llama3:8b
ollama list
ollama serve          # or run as a service / in another terminal
```

### Smoke checks

```bash
curl http://localhost:11434/api/tags
ollama run llama3:8b "Hello, how are you?"
```

### Use in code

```python
from llm_client_selection import OllamaLLMClient

client = OllamaLLMClient(
    model_name="llama3:8b",
    base_url="http://localhost:11434",
)
```

Typical generation options (when exposed by your caller): temperature ~0.7, top_p ~0.9, max tokens and timeout tuned for your hardware.

### Fallback behavior

- Ollama down or model missing → **OllamaLLMSelector** uses the **adaptive** selector instead.
- Retries with exponential backoff on timeouts; parse failures may fall back to safer selection strategies.

### Troubleshooting Ollama

- Service not running: `ollama serve`, check port **11434** (`ss -tlnp | grep 11434` or `netstat`).
- Model missing: `ollama pull llama3:8b`.
- OOM / slow: try a smaller tag (e.g. `llama3:8b` vs larger variants) or reduce prompt size and timeouts.

---

## Loss weighting agent (`loss_weighting_agent.py`)

### Idea

Instead of a fixed scalar on the auxiliary batch term, the agent proposes a **bounded** coefficient per client using context (global loss/accuracy, per-client losses, weights, convergence, diversity, stability, etc.), optionally via **Ollama**; otherwise **rule-based** weighting.

### Core types

- **`LossWeightingContext`**: round index, global metrics, per-client losses/accuracies/weights, convergence, diversity, prior effectiveness, stability, communication efficiency.
- **`WeightingStrategy`**: `base_coefficient`, `adaptive_factor`, `diversity_factor`, `convergence_factor`, `stability_factor`, `exploration_rate`.

### Example

```python
from loss_weighting_agent import (
    create_loss_weighting_agent,
    LossWeightingContext,
)

agent = create_loss_weighting_agent(model_name="llama3:8b", logger=logger)

context = LossWeightingContext(
    current_round=epoch_index,
    total_rounds=Config.COMMUNICATION_EPOCH,
    global_loss=global_loss,
    global_accuracy=global_accuracy,
    client_losses=client_losses,
    client_accuracies=client_accuracies,
    client_weights=client_weights,
    convergence_rate=convergence_rate,
    diversity_score=diversity_score,
    previous_weighting_effectiveness=previous_effectiveness,
    training_stability=training_stability,
    communication_efficiency=communication_efficiency,
)
coef = agent.calculate_loss_weighting(context, client_index)
# loss = loss + coef * w * loss_batch_sample  (as in main training loop)
agent.update_effectiveness(old_coef, coef, performance_improvement)
stats = agent.get_weighting_statistics()
```

### Tests

```bash
python test_loss_weighting_agent.py
```

---

## Client aggregation weights (`weight_quality`)

`main.py` treats **`weight_quality`** as a **learnable** vector (length `N_CLIENTS`, nonnegative, **sum to 1** after a softmax step) updated each communication round when `epoch_index > 0`:

1. **Quality signal** from relative loss change per client vs the previous round.
2. **Normalized** quality targets.
3. **Optimization**: maximize alignment of `weight_quality` with normalized quality (negative dot product as loss) plus **L2-style regularization** toward uniform weights (strength configurable in code, historically `0.1`).
4. **Adam** on `weight_quality` (learning rate often `0.01` in code).
5. **Softmax** on the parameter tensor so weights stay a simplex.

When applying these weights in the loss, values are typically **`.detach()`** so downstream gradients do not flow back through the weight-quality update incorrectly.

Tunable knobs (in `main.py`): optimizer learning rate, regularization coefficient, normalization choice (softmax vs alternatives).

---

## CIFAR-10 utilities (`Dataset/`)

| File | Role |
|------|------|
| `cifar10_to_images.py` | Pickle batches → `train/<class>/`, `test/<class>/` images. |
| `cifar10_dataset.py` | PyTorch `Dataset`, optional `dataidxs`, transforms. |

```python
from Dataset.cifar10_dataset import CIFAR10Dataset

ds = CIFAR10Dataset(root="data/cifar", train=True)
print(len(ds), ds.class_names)
```

---

## Logging and checkpoints

- **Logs**: `Logs/` (log file name follows the script being run).
- **Models**: `Model_Storage/` (per-client or round naming as implemented in training scripts).

---

## General troubleshooting

| Issue | What to check |
|-------|----------------|
| Dataset not found | Paths in `config.py`; GenImage subfolder names vs `DATASET_LIST`. |
| Wrong GenImage layout | Must have `train/nature`, `train/ai`, `val/nature`, `val/ai`. |
| Empty CIFAR tree | Run `Dataset/cifar10_to_images.py` from `Dataset/`; confirm `data/cifar/`. |
| CUDA OOM | Reduce batch sizes in `config.py`. |
| Ollama / LLM errors | `ollama serve`, `ollama pull`, firewall on `11434`; agents fall back to non-LLM logic when possible. |
| Import errors | Run from repo root; `Dataset` package and `sys.path` usage in scripts. |

---

## Key project files

| File | Purpose |
|------|---------|
| `main.py` | Hybrid FL, `weight_quality`, LLM selection, loss weighting. |
| `config.py` | Dataset type, paths, nets, hyperparameters. |
| `pretrain.py` | Pretraining entry point. |
| `train_fedavg.py` | Standalone FedAvg. |
| `Dataset/init_dataset.py` | `genimage`, `celebdf2` datasets. |
| `Dataset/utils.py` | Dataloaders, **`init_nets`** (EfficientNet-B0 default), logging helpers. |
| `loss.py` | CE, SCE, etc. |
| `llm_client_selection.py` | Selection strategies and Ollama client. |
| `loss_weighting_agent.py` | Dynamic loss coefficients. |

---

## License and third-party code

Respect the licenses of **GenImage**, **CIFAR-10**, and **Celeb-DF** (and any other datasets you add) when redistributing or publishing results.
