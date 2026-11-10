# SmartFed

SmartFed is a federated learning codebase focused on deepfake detection and related benchmarks. It supports two FL **protocols** (homogeneous vs heterogeneous client backbones), **GenImage** (binary real/fake, non-IID across clients), **CIFAR-10**, and related utilities. The main training entry point is `main.py`; a **FedAvg-only** script is `train_fedavg.py`.

This file is the project documentation: setup, data layout, Ollama/LLM features, and the experiment runners under `scripts/`.

Datasets, logs, and checkpoints are **not** in the git repository. Clone the code, place data on disk, then run `pretrain.py` before `main.py`. Paths and round counts are overridden with `SMARTFED_*` environment variables (see [CIFAR-10 agent modes](#cifar-10-client-selection-and-loss-weighting)); `config.py` only chooses GenImage (`config_genimage.py`) or CIFAR-10 (`config_cifar.py`).

---

## Features

- **Protocols** (`config.py` → **`PROTOCOL`**): **`Homo_FL`**, **`Hete_FL`**, or **`Homo_ServerGSAM`** (GSAM also updates server party 0; see [FL protocols](#fl-protocols-homo_fl--hete_fl--homo_servergsam)).
- **Backbones**: EfficientNet-B0 (`torchvision`), plus ResNet10 / ResNet12 / MobileNetV2 for Hete_FL (`Dataset/utils.py` → `init_nets`). GenImage-style inputs are **224×224**.
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

## FL protocols (`Homo_FL` / `Hete_FL` / `Homo_ServerGSAM`)

Set **`PROTOCOL`** near the top of `Config` in `config.py`. This chooses how **`PRIVATE_NETS_NAME_LIST`** / **`NETS_NAME_LIST`** are filled (after **`N_CLIENTS`** is known from the dataset branch), and whether GSAM updates the server.

| Protocol | Meaning | Backbones | GSAM parties |
|----------|---------|-----------|--------------|
| **`Homo_FL`** (default) | Homogeneous FL | All parties: **`EfficientNetB0`** | Clients **1..N** only |
| **`Hete_FL`** | Heterogeneous FL | Server: EfficientNetB0; clients round-robin **`HETE_BACKBONE_POOL`** | Clients **1..N** only |
| **`Homo_ServerGSAM`** | Homogeneous + server local update | All parties: **`EfficientNetB0`** | **0** (on `genimage_pub`) **and** clients **1..N** |

### Party layout (GenImage / DiffusionFace)

With **`HAS_SERVER_TEACHER = True`** (default for GenImage):

| Party id | Role | Data |
|----------|------|------|
| **0** | **Server teacher** (shared at server) | `PUBLIC_DATASET_DIR` → `genimage_pub` |
| **1 .. N_CLIENTS** | Private **clients** | `DATASET_LIST[0 .. N_CLIENTS-1]` (e.g. `adm` … `wukong`) |

So for GenImage: **`N_CLIENTS = 8`**, **`N_PARTIES = 9`**, checkpoints `EfficientNetB0_0.ckpt` … `_8.ckpt`.  
Public KL uses party **0** as the frozen teacher during the KL window; under **`Homo_FL` / `Hete_FL`**, GSAM / local updates run on parties **1–8** only. Under **`Homo_ServerGSAM`**, GSAM also fine-tunes party **0** on `genimage_pub` every round (after KL).

**Communication topology (star only):** each round the server updates with **one client at a time** — pairs `(0,1), (0,2), …, (0,8)`. Soft labels and weight/KL updates never go client↔client (e.g. no `1↔2`).

### Assignment example (GenImage, `N_CLIENTS = 8` → parties 0–8)

**Homo_FL** / **Homo_ServerGSAM:**

```text
party 0 (server teacher, genimage_pub) → EfficientNetB0
party 1..8 (clients adm..wukong)     → EfficientNetB0
```

**Hete_FL** (server teacher fixed to EfficientNetB0; clients round-robin the pool):

```text
party 0 → EfficientNetB0   # server teacher
party 1 → EfficientNetB0   # adm
party 2 → ResNet10
party 3 → ResNet12
party 4 → Mobilenetv2
party 5 → EfficientNetB0
party 6 → ResNet10
party 7 → ResNet12
party 8 → Mobilenetv2
```

Customize the cycle by editing **`HETE_BACKBONE_POOL`** in `config.py` (names must match `init_nets` in `Dataset/utils.py`).

### How to run each protocol

1. Set **`PROTOCOL = 'Homo_FL'`**, **`'Hete_FL'`**, or **`'Homo_ServerGSAM'`** in `config.py`.
2. Keep **`DATASET_TYPE`**, paths, and **`N_CLIENTS`** as usual.
3. Run **`pretrain.py`** with that protocol so each **party** writes the matching checkpoint, e.g.  
   `Model_Storage/CE/None0/EfficientNetB0_0.ckpt` (server teacher on `genimage_pub`), `…_1.ckpt` (first client), …
4. Run **`main.py`** (or **`train_fedavg.py`**) with the **same** **`PROTOCOL`** so load/save names align.

Switching protocol without re-pretraining will look for the wrong `.ckpt` filenames.  
**Re-pretrain after this party-layout change** — old `*_0.ckpt` was `adm`, now `*_0.ckpt` is the server teacher on `genimage_pub`.  
`Homo_FL` ↔ `Homo_ServerGSAM` share the same checkpoint names (both EfficientNetB0); only the GSAM party set differs at runtime.

Public KL distillation still works across architectures because alignment is on **logits** (same `OUTPUT_CHANNEL`), not shared weights.

---

## Quick start

1. **Configure `config.py`** — set **`PROTOCOL`**, then follow [Declaring `config.py` for a successful run](#declaring-configpy-for-a-successful-run) so dataset paths, client count, and model settings match your machine and data layout.
2. Prepare data on disk as in [Dataset layout on disk](#dataset-layout-on-disk) (paths must match what you set under `PRIVATE_DATASET_DIR` / `PUBLIC_DATASET_DIR`).
3. Pretraining (required for `main.py` with matching protocol):

   ```bash
   python pretrain.py
   ```

4. Main training:

   ```bash
   python main.py
   ```

5. FedAvg-only:

   ```bash
   python train_fedavg.py
   ```

---

## Declaring `config.py` for a successful run

All training scripts (`main.py`, `pretrain.py`, `train_fedavg.py`) read the **`Config`** class. `config.py` loads **`config_genimage.py`** by default and **`config_cifar.py`** when `SMARTFED_CONFIG=cifar` or `SMARTFED_DATASET_TYPE=cifar10`. Most experiment knobs (`PROTOCOL`, `N_CLIENTS`, `COMMUNICATION_EPOCH`, `WEIGHT_RULE`, dataset directories, agent mode) are `SMARTFED_*` environment variables with the defaults in those two files. Edit a file only for a default you want to keep.

### 1. How `config.py` is structured

- **`_SMARTFED_ROOT`**: absolute path to the SmartFed repository (the folder that contains `config.py`). Dataset directories default to **`os.path.join(_SMARTFED_ROOT, 'data', ...)`**.
- **`Config`**: a class whose body runs top to bottom. **`DATASET_TYPE`** selects **one** branch (`'genimage'` or `'cifar10'`). Only the variables in that branch are defined for dataset-specific paths and names.

To use data **outside** the repo, replace the `os.path.join(_SMARTFED_ROOT, ...)` expressions in the active branch with your own absolute paths (or define a variable like `MY_DATA = '/mnt/datasets'` and join from that).

---

### 2. Minimal checklist before the first run

| Step | What to verify |
|------|----------------|
| Protocol | Set **`PROTOCOL`** to `'Homo_FL'`, `'Hete_FL'`, or `'Homo_ServerGSAM'` (see [FL protocols](#fl-protocols-homo_fl--hete_fl--homo_servergsam)). |
| Dataset mode | Set **`DATASET_TYPE`** to `'genimage'`, `'diffusionface'`, or `'cifar10'`. |
| Data on disk | Folders exist and match [Dataset layout on disk](#dataset-layout-on-disk) for the chosen mode. |
| Paths | **`PRIVATE_DATASET_DIR`** and **`PUBLIC_DATASET_DIR`** (in the active `if` / `elif` branch) point to those folders. |
| Client count | **`N_CLIENTS`** equals the number of clients you train; for GenImage it must match **`len(DATASET_LIST)`**. |
| Classes vs model | **`OUTPUT_CHANNEL`** and **`PRIVATE_OUTPUT_CHANNEL`** equal `len(PRIVATE_DATASET_CLASSES)` (2 for GenImage, 10 for CIFAR-10 as shipped). |
| Network names | **`PRIVATE_NETS_NAME_LIST`** / **`NETS_NAME_LIST`** are filled from **`PROTOCOL`** (length **`N_CLIENTS`**). Resolved in `Dataset/utils.py` → `init_nets`. |
| `main.py` checkpoints | **`main.py`** expects per-client checkpoints under **`Model_Storage/...`** (see §7). Run **`pretrain.py`** first with the **same `PROTOCOL`**, or place `.ckpt` files manually. **`train_fedavg.py`** starts from freshly initialized nets (no `Model_Storage` load unless you change the code). |

---

### 3. Global training and federated parameters

These apply regardless of dataset (see `config.py` for current numeric defaults).

| Variable | Meaning |
|----------|---------|
| **`PROTOCOL`** | `'Homo_FL'`, `'Hete_FL'`, or `'Homo_ServerGSAM'` — backbone layout + whether GSAM updates the server (see [FL protocols](#fl-protocols-homo_fl--hete_fl--homo_servergsam)). |
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

### 6. Model architecture lists (driven by `PROTOCOL`)

Do **not** hand-edit **`PRIVATE_NETS_NAME_LIST`** / **`NETS_NAME_LIST`** unless you know what you are doing — they are set automatically from **`PROTOCOL`**:

| Variable | Meaning |
|----------|---------|
| **`PROTOCOL`** | `'Homo_FL'` → all **`EfficientNetB0`**. `'Hete_FL'` → round-robin over **`HETE_BACKBONE_POOL`**. |
| **`HETE_BACKBONE_POOL`** | Backbone names used only when **`PROTOCOL == 'Hete_FL'`**. Default: `EfficientNetB0`, `ResNet10`, `ResNet12`, `Mobilenetv2`. |
| **`PRIVATE_NETS_NAME_LIST`** | Length **`N_CLIENTS`**: per-client architecture and checkpoint filename stem. |
| **`NETS_NAME_LIST`** | Copy of **`PRIVATE_NETS_NAME_LIST`**; used by `pretrain.py` / `train_fedavg.py` when calling `init_nets`. |

Supported names in **`Dataset/utils.py`** → **`init_nets`**: `EfficientNetB0`, `EfficientNetB4`, `ResNet10`, `ResNet12`, `Mobilenetv2` (aliases `MobileNetV2` / `MobileNetv2`), `ShuffleNet`.

---

### 7. Checkpoints and `Model_Storage` layout (`pretrain.py` / `main.py`)

**`pretrain.py`** writes:

- Per-client:  
  `Model_Storage/<PARTICIPANT_PARAMS['loss_function']>/<NOISE_TYPE><NOISE_RATE>/<PRIVATE_NETS_NAME_LIST[i]>_<i>.ckpt`  
  Examples: `EfficientNetB0_0.ckpt` (Homo_FL), or `ResNet10_1.ckpt` / `Mobilenetv2_3.ckpt` (Hete_FL).
- Optional public model path used in pretrain:  
  `.../public_model.ckpt` under the same `<loss>/<noise>` folder.

**`main.py`** loads each client’s checkpoint from the **same pattern** before continuing training. So for a successful **`main.py`** run you must either:

1. Run **`pretrain.py`** first with the **same** `PROTOCOL`, `PARTICIPANT_PARAMS`, `NOISE_TYPE`, `NOISE_RATE`, and `N_CLIENTS`, or  
2. Place compatible `.ckpt` files at those paths manually.

If any path segment does not match, you will get a **file not found** error at startup.

---

### 8. When `DATASET_TYPE == 'genimage'`

Set **`DATASET_TYPE = 'genimage'`** at the top of the dataset block. The following symbols are then active:

| Variable | You must ensure |
|----------|------------------|
| **`DATASET_LIST`** | List of **subdirectory names** under **`PRIVATE_DATASET_DIR`**, one per **client**, length **`N_CLIENTS`**. Default: eight generator folders (`adm`, `BigGAN`, …). Each must contain `train/nature`, `train/ai`, `val/nature`, `val/ai`. Mapped to parties **1 .. N_CLIENTS**. |
| **`PRIVATE_DATASET_DIR`** | Parent directory of all client folders. Default: `<repo>/data/genimage_medium_fulltest`. |
| **`PRIVATE_DATASET_NAME`** | Must stay **`'genimage'`** for the current `genimage` loader. |
| **`PRIVATE_DATA_LEN`** | Used when **`INTRA_DATASET`** is `True` (cap on indices per client). For `False`, behavior depends on full dataset size. |
| **`PRIVATE_DATASET_CLASSES`** | **`['real', 'fake']`** → binary classification; must match **`OUTPUT_CHANNEL` / `PRIVATE_OUTPUT_CHANNEL`**. |
| **`OUTPUT_CHANNEL`**, **`PRIVATE_OUTPUT_CHANNEL`** | Set to **`len(PRIVATE_DATASET_CLASSES)`** (2). |
| **`PUBLIC_DATASET_NAME`** | **`'genimage'`** for the public / server-teacher loader. |
| **`PUBLIC_DATASET_DIR`** | **`…/genimage_pub`** — data for **party 0 (server teacher)** and for Public KL soft targets. Same folder layout as a client root. |
| **`PUBLIC_DATASET_LENGTH`** | Size hint / subsampling for public data where the training code uses it (default `8000`). |
| **`HAS_SERVER_TEACHER`** | **`True`**: party 0 = server teacher; **`N_PARTIES = N_CLIENTS + 1`**. |
| **`CLIENT_IDS`** | e.g. **`[1..8]`** — parties that run local weight / GSAM / across-eval. |

**Consistency:** **`len(DATASET_LIST) == N_CLIENTS`**. Checkpoints: **`N_PARTIES`** files (`*_0.ckpt` = teacher, `*_1.ckpt`… = clients).

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
3. **`PROTOCOL`** changed after pretrain (or mismatched Homo vs Hete) → wrong checkpoint filenames (`EfficientNetB0_i` vs `ResNet10_i`, …).  
4. **`main.py`** without pretrain → missing **`Model_Storage/...`** files.  
5. **`NOISE_TYPE`** / **`NOISE_RATE`** changed after pretrain → checkpoint path no longer matches.  
6. Data not under `<repo>/data/...` → update the **`os.path.join(_SMARTFED_ROOT, ...)`** lines in the active **`DATASET_TYPE`** branch.

After any change to **`PROTOCOL`** or backbones, re-run **`pretrain.py`**, then **`main.py`** or **`train_fedavg.py`**.

---

## Dataset layout on disk

Training expects **fixed subdirectory names** unless you change `Dataset/init_dataset.py` or `Dataset/cifar10_dataset.py`.

### GenImage (`DATASET_TYPE = 'genimage'`)

**Server teacher (party 0):** `PUBLIC_DATASET_DIR` (e.g. `.../genimage_pub`).  
**Clients (parties 1..N):** `PRIVATE_DATASET_DIR / <name>` via `Config.party_datadir(party_id)`.

Under each root:

```text
<root>/
├── train/
│   ├── nature/     # real → label 0
│   └── ai/         # fake → label 1
└── val/
    ├── nature/
    └── ai/
```

**Known data bug (fixed by script):** `wukong/val/ai` was a byte-identical copy of `vqdm/val/ai` (`VQDM_*` files) while `wukong/train/ai` was correct. That made wukong own-val ≈50% and forced `vqdm ≡ wukong` metrics. Repair with:

```bash
python scripts/fix_wukong_val.py          # rebuild val/ai from wukong train holdout
# then re-pretrain party 8 only (Config.PRETRAIN_PARTY_IDS = [8])
python pretrain.py
```

`main.py` / `pretrain.py` call `Dataset.integrity.check_genimage_client_val` and will **abort** if the bad VQDM-in-wukong-val pattern reappears.

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

## CIFAR-10: client selection and loss weighting

Three agent settings, from no learned agents to a fixed rule to an LLM, plus a log of LLM calls, latency, and fallbacks.

Run all commands from the repo root.

### Modes (`config_cifar.py` / env)

Use **`SMARTFED_CONFIG=cifar`** (or `SMARTFED_DATASET_TYPE=cifar10`) so `config.py` loads `config_cifar.py` instead of GenImage.

| Mode | Env | DCSA | AFWA |
|------|-----|------|------|
| No agents | `SMARTFED_AGENT_MODE=wo_ma` | all clients (budget=16) | fixed coef `0.1` |
| Rule-based | `SMARTFED_AGENT_MODE=rule_based` | `adaptive` (no LLM), all clients | rule-based coefficients (`use_llm=False`) |
| LLM | `SMARTFED_AGENT_MODE=llm` | Ollama Llama3, all clients | Ollama Llama3, \(\alpha\in[0.05,0.3]\) |

To change selection and weighting separately, set `SMARTFED_AGENT_MODE=custom` and override `SMARTFED_DCSA_MODE` / `SMARTFED_AFWA_MODE` ∈ `{off, rule_based, llm}`.

CIFAR-10 defaults in `config_cifar.py`: **16 clients**, `Homo_FL`, `GSAM_TOTAL_EPOCHS=5`, KL **head-only** / 20 batches / \(T=2\), KL teacher = `public_model.ckpt`, selection budget **16** (the selection sweep below uses **4**), checkpoints under `Model_Storage_cifar10/`.

Official pickles (optional convert): `data/cifar-10-python`

Cost instrumentation (`llm_client_selection.GLOBAL_LLM_COST`): latency per call, calls/round, tokens when Ollama returns them, fallback rate (timeout / API / empty / parse / selector). Logged as `phase=llm_cost` in the metrics CSV and `LLM cost cumulative` at end of `main.py`.

### Suggested order

1. (Optional) Convert CIFAR pickles → images  
2. Pretrain seeds `0,1,2`  
3. Agent modes: `wo_ma` / `rule_based` / `llm` × 3 seeds  
4. Selection sweep: vary DCSA, AFWA stays on the LLM × 3 seeds  
5. Weighting sweep: vary AFWA, DCSA stays on the LLM × 3 seeds  
6. Summarize logs  

For LLM modes: `ollama serve` and `ollama pull llama3:8b`.

### Batch runner (optional)

```bash
# All three agent modes, seeds 0, 1, and 2
python scripts/run_r3_major4_cifar.py --seeds 0,1,2 --figure both

# Selection sweep only (AFWA stays on the LLM) / weighting sweep only (DCSA stays on the LLM)
python scripts/run_r3_major4_cifar.py --seeds 0,1,2 --figure 4
python scripts/run_r3_major4_cifar.py --seeds 0,1,2 --figure 5

# LLM mode only: 8 clients, public set 6,000 train / 1,040 test
# (the remaining images are split equally; logs go under Logs/r3_major4/<stamp>)
python scripts/run_r3_major4_cifar.py --seeds 0,1,2 --figure llm \
  --n-clients 8 --public-train 6000 --public-test 1040

# Mean accuracy plus LLM call count, latency, and fallback rate
python scripts/summarize_r3_major4.py --log-dir Logs/r3_major4/<stamp>
```

### 0) Convert pickle → images (skip if `data/cifar` already exists)

```bash
python scripts/convert_cifar10.py \
  --pickle-root data/cifar-10-python \
  --output-dir data/cifar
```

### 1) Pretrain (one run per seed; shared by all modes)

**Seed 0**

```bash
SMARTFED_CONFIG=cifar \
SMARTFED_N_CLIENTS=16 \
SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python pretrain.py
```

**Seed 1** — same command with `SMARTFED_SEED=1`  
**Seed 2** — same command with `SMARTFED_SEED=2`

### 2) Agent modes: none, rule-based, LLM

Repeat each block below with `SMARTFED_SEED=0`, then `1`, then `2`.

#### 2.1 No agents

```bash
SMARTFED_CONFIG=cifar \
SMARTFED_N_CLIENTS=16 \
SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 \
SMARTFED_AGENT_MODE=wo_ma \
SMARTFED_SELECTION_BUDGET=16 \
SMARTFED_COMMUNICATION_EPOCH=40 \
SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

#### 2.2 Rule-based — no LLM calls

```bash
SMARTFED_CONFIG=cifar \
SMARTFED_N_CLIENTS=16 \
SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 \
SMARTFED_AGENT_MODE=rule_based \
SMARTFED_SELECTION_BUDGET=16 \
SMARTFED_COMMUNICATION_EPOCH=40 \
SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

#### 2.3 LLM-driven (Llama3:8b)

```bash
SMARTFED_CONFIG=cifar \
SMARTFED_N_CLIENTS=16 \
SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 \
SMARTFED_AGENT_MODE=llm \
SMARTFED_OLLAMA_MODEL=llama3:8b \
SMARTFED_SELECTION_BUDGET=16 \
SMARTFED_COMMUNICATION_EPOCH=40 \
SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

### 3) Selection sweep (weighting stays on the LLM)

Use **budget 4** so selection has an effect. With selection off, 4 clients are sampled at random.

#### 3.1 DCSA off

```bash
SMARTFED_CONFIG=cifar SMARTFED_N_CLIENTS=16 SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 SMARTFED_AGENT_MODE=custom \
SMARTFED_DCSA_MODE=off SMARTFED_AFWA_MODE=llm \
SMARTFED_OLLAMA_MODEL=llama3:8b SMARTFED_SELECTION_BUDGET=4 \
SMARTFED_COMMUNICATION_EPOCH=40 SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

#### 3.2 DCSA rule-based (no LLM)

```bash
SMARTFED_CONFIG=cifar SMARTFED_N_CLIENTS=16 SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 SMARTFED_AGENT_MODE=custom \
SMARTFED_DCSA_MODE=rule_based SMARTFED_DCSA_RULE_STRATEGY=adaptive \
SMARTFED_AFWA_MODE=llm \
SMARTFED_OLLAMA_MODEL=llama3:8b SMARTFED_SELECTION_BUDGET=4 \
SMARTFED_COMMUNICATION_EPOCH=40 SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

#### 3.3 DCSA LLM

```bash
SMARTFED_CONFIG=cifar SMARTFED_N_CLIENTS=16 SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 SMARTFED_AGENT_MODE=custom \
SMARTFED_DCSA_MODE=llm SMARTFED_AFWA_MODE=llm \
SMARTFED_OLLAMA_MODEL=llama3:8b SMARTFED_SELECTION_BUDGET=4 \
SMARTFED_COMMUNICATION_EPOCH=40 SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

### 4) Weighting sweep (selection stays on the LLM)

#### 4.1 AFWA off (fixed coefficient 0.1)

```bash
SMARTFED_CONFIG=cifar SMARTFED_N_CLIENTS=16 SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 SMARTFED_AGENT_MODE=custom \
SMARTFED_DCSA_MODE=llm SMARTFED_AFWA_MODE=off \
SMARTFED_AFWA_FIXED_COEFFICIENT=0.1 \
SMARTFED_OLLAMA_MODEL=llama3:8b SMARTFED_SELECTION_BUDGET=16 \
SMARTFED_COMMUNICATION_EPOCH=40 SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

#### 4.2 AFWA rule-based

```bash
SMARTFED_CONFIG=cifar SMARTFED_N_CLIENTS=16 SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 SMARTFED_AGENT_MODE=custom \
SMARTFED_DCSA_MODE=llm SMARTFED_AFWA_MODE=rule_based \
SMARTFED_OLLAMA_MODEL=llama3:8b SMARTFED_SELECTION_BUDGET=16 \
SMARTFED_COMMUNICATION_EPOCH=40 SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

#### 4.3 AFWA LLM

```bash
SMARTFED_CONFIG=cifar SMARTFED_N_CLIENTS=16 SMARTFED_PROTOCOL=Homo_FL \
SMARTFED_SEED=0 SMARTFED_AGENT_MODE=custom \
SMARTFED_DCSA_MODE=llm SMARTFED_AFWA_MODE=llm \
SMARTFED_OLLAMA_MODEL=llama3:8b SMARTFED_SELECTION_BUDGET=16 \
SMARTFED_COMMUNICATION_EPOCH=40 SMARTFED_LOG_LLM_COST=1 \
SMARTFED_PRIVATE_DATASET_DIR=data/cifar \
SMARTFED_PUBLIC_DATASET_DIR=data/cifar \
python main.py
```

### 5) Summarize results

```bash
# After the batch runner:
python scripts/summarize_r3_major4.py --log-dir Logs/r3_major4/<stamp>

# After manual runs: read accuracy / "LLM cost cumulative" in Logs/main_*.log
# and phase=llm_cost rows in Logs/main_metrics_*.csv
```

---

## General troubleshooting

| Issue | What to check |
|-------|----------------|
| Dataset not found | Paths in `config.py`; GenImage subfolder names vs `DATASET_LIST`. |
| Wrong GenImage layout | Must have `train/nature`, `train/ai`, `val/nature`, `val/ai`. |
| Empty CIFAR tree | Run `scripts/convert_cifar10.py` (or `Dataset/cifar10_to_images.py`); confirm `data/cifar/`. |
| CUDA OOM | Reduce batch sizes in `config.py`. |
| Ollama / LLM errors | `ollama serve`, `ollama pull`, firewall on `11434`; agents fall back to non-LLM logic when possible. |
| Import errors | Run from repo root; `Dataset` package and `sys.path` usage in scripts. |

---

## Key project files

| File | Purpose |
|------|---------|
| `main.py` | Hybrid FL, `weight_quality`, LLM selection, loss weighting. |
| `config.py` | Dispatcher: GenImage by default, CIFAR when `SMARTFED_CONFIG=cifar`. |
| `config_cifar.py` | CIFAR-10 hyperparameters (16 clients, GSAM=5, KL head-only, AFWA clip). |
| `config_genimage.py` | GenImage / DiffusionFace hyperparameters. |
| `pretrain.py` | Pretraining entry point. |
| `train_fedavg.py` | Standalone FedAvg. |
| `Dataset/init_dataset.py` | `genimage`, `celebdf2` datasets. |
| `Dataset/utils.py` | Dataloaders, **`init_nets`** (Homo/Hete backbones), logging helpers. |
| `loss.py` | CE, SCE, etc. |
| `llm_client_selection.py` | Selection strategies, Ollama client, **`LLMCostTracker`**. |
| `loss_weighting_agent.py` | Dynamic loss coefficients (LLM or a fixed rule). |
| `scripts/run_r3_major4_cifar.py` | Multi-seed CIFAR runner for the agent modes above. |
| `scripts/summarize_r3_major4.py` | Aggregate accuracy and LLM cost from those logs. |
| `scripts/run_r2_table7_genimage.py` | GenImage comparison of uniform, dataset-size, and signed-loss \(w_k\) (20 rounds each). |
| `scripts/convert_cifar10.py` | Pickle → `data/cifar` images. |

Checkpoints (`*.ckpt`), `Logs/`, `Model_Storage*/`, and `data/` are gitignored. A fresh clone has source only.

---

## License and third-party code

Respect the licenses of **GenImage**, **CIFAR-10**, and **Celeb-DF** (and any other datasets you add) when redistributing or publishing results.
