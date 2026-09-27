# ECFNet

ECFNet fuses fixed wavelet descriptors with band-specific regional graph representations for EEG emotion recognition. This repository contains the model, SEED preprocessing, subject-wise evaluation, ablation configurations and implementation tests.

[Data preparation](docs/data.md) · [Experiments and evaluation](docs/experiments.md) · [中文项目结构说明](docs/architecture.md)

## Repository structure

```text
ECFNet/
├── main.py                 # Training, data preparation and evaluation entry point
├── cross_validation.py     # Subject splits and experiment orchestration
├── datapipe.py              # SEED loading and feature caches
├── dataset.py              # Dataset wrappers
├── train.py                # Training, early stopping and checkpoints
├── visualize.py            # Optional model interpretation
├── net/                    # ECFNet, shared layers and baseline adapters
├── utils/                  # Features, channel partitions and configuration tools
├── config/
│   ├── index.yaml          # ECFNet-G with historical evaluation settings
│   ├── index-strict.yaml   # Independent subject validation
│   ├── ablations/          # Structural and fusion ablations
│   └── legacy/             # Earlier development configurations
├── docs/                   # Data, experiment and architecture documentation
├── tests/                  # Formula and pipeline checks
├── scripts/                # Source-release packaging
└── .github/workflows/      # Automated implementation checks
```

## Installation

Use Python 3.10–3.12 and run commands from the repository root.

```bash
uv sync --frozen --extra test
uv run --frozen --extra test python -m pytest -q
```

Alternatively, create and activate a virtual environment, then install with pip:

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python -m pip install 'pytest>=8,<10'
python -m pytest -q
```

`uv.lock` pins the resolved environment. PyTorch selects CUDA when available and CPU otherwise. Set `training.device` in the configuration to select a device explicitly. Optional EEG interpretation dependencies are provided by the `visualization` extra.

## Prepare data and run

Obtain the 200-Hz, 62-channel SEED preprocessed recordings and place them under `data/raw/SEED/Preprocessed_EEG/`, or override `dataset.raw_root`. See [data preparation](docs/data.md) for the required files and preprocessing details.

```bash
# Prepare the feature cache.
uv run --frozen python main.py --config config/index-strict.yaml --prepare-only

# Train all held-out-subject folds.
uv run --frozen python main.py --config config/index-strict.yaml

# Re-evaluate saved fold checkpoints.
uv run --frozen python main.py --config config/index-strict.yaml --evaluate
```

`index-strict.yaml` trains on 13 subjects, uses one separate subject for validation and reserves one subject for testing. `index.yaml` retains the historical 14-subject training protocol with test-based epoch selection. Results from these protocols should be reported separately. [Experiment documentation](docs/experiments.md) describes normalization, ablations, partial-fold runs and checkpoint evaluation.

Outputs are written under `run/`, separated by `training.exp_name`; feature caches are stored under `data/processed/SEED/<config_name>/`. These generated files are excluded from version control. Use a new experiment name for a new run.

## Ablations and interpretation

```bash
uv run --frozen python main.py --config config/ablations/shared_encoder.yaml

uv sync --frozen --extra visualization
uv run --frozen --extra visualization python visualize.py --config config/index-strict.yaml --subject 1
```

The ablation configs cover regional partitions, descriptor removal, encoder sharing, fusion strategies and graph propagation depth. Earlier configurations are retained in `config/legacy/` for reference.

## Python interface

```python
import torch
from net import ECFNet

model = ECFNet(num_bands=5, num_classes=3,
               area_nodes=[3, 2, 9, 7, 7, 7, 9, 7, 5, 3, 3]).eval()
de = torch.randn(2, 37, 62, 5)  # electrodes must be reordered into the chosen regions
wavelet = torch.randn(2, 37, 3)
with torch.no_grad():
    logits = model((de, wavelet))
    probabilities = logits.softmax(dim=-1)
```

The repository uses `(batch, windows, channels, bands)` for DE, equivalent to the manuscript's `(T,B,C)` notation after axis reordering. No positional embeddings are used. During evaluation, segment predictions are invariant to independent window permutations in the two streams.

