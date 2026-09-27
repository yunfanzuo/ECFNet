# Experiment configurations and evaluation

## Evaluation protocols

Two explicitly named protocols are provided:

| Config | Training subjects per fold | Epoch selection | Normalization |
| --- | --- | --- | --- |
| `config/index.yaml` | 14 | Held-out test accuracy (`legacy_test`) | Per-subject statistics, including the held-out subject |
| `config/index-strict.yaml` | 13 | One independent training-side validation subject (`validation_subject`) | Fit on the 13 training subjects, apply to validation and test |

`index.yaml` preserves the supplied source's historical selection and normalization choices while using the corrected ECFNet implementation. It uses test labels to choose the best epoch, so its score is **not an independent test estimate**. The PDF specifies LOSO and early stopping but does not specify the early-stopping split or normalization. Do not describe these source defaults as paper-stated settings.

**Use the independent validation configuration for new evaluations:**

```bash
uv run --frozen python main.py --config config/index-strict.yaml --prepare-only
uv run --frozen python main.py --config config/index-strict.yaml
```

For test subject `s`, the next subject in sorted order is used for validation, wrapping from 15 to 1. The remaining 13 subjects train the model. There is no refitting on all 14 subjects. Test metrics are computed once after restoring the validation-selected checkpoint. This is a separate evaluation protocol and must be reported separately from the paper's historical 14-subject training setup.

Strict normalization standardizes each DE electrode across training samples, windows and bands; each descriptor is standardized separately across training samples and windows. These choices are explicit implementation settings, since the PDF omits normalization details.

Both configs retain paper-stated Adam (learning rate 0.001), batch size 64, maximum 100 epochs, patience 10, K=2, and label smoothing 0.05. Dropout probability 0.5 and base seed 42 are inherited source defaults, not specified in the paper. Fold seed is `base_seed + subject_id`, so a selected fold can be run independently. Set `reproduce.deterministic: true` to request deterministic PyTorch algorithms; this may slow training or reject unsupported operations.

To run selected held-out subjects:

```bash
uv run --frozen python main.py --config config/index-strict.yaml --folds 1 2
```

The training pool still contains all eligible subjects; `--folds` only selects test folds. A subset is marked `complete_loso: false`. Use a new `training.exp_name` and `logging.save_dir` for a fresh experiment. Completed folds cannot be overwritten accidentally. You can add uncompleted folds under the same unchanged configuration.

## Ablations

Each supplied ablation inherits **independent validation** from `index-strict.yaml`. Configs instantiate the paper's structural comparisons; they do not embed published scores.

| Paper comparison | Config under `config/ablations/` |
| --- | --- |
| ECFNet-F, 14 regions | `frontal.yaml` |
| ECFNet-H, 19 regions | `hemisphere.yaml` |
| Without local pooling (62 singleton regions) | `no_local_pooling.yaml` |
| Without descriptors (graph projection and self-attention retained) | `no_descriptors.yaml` |
| Shared band encoder | `shared_encoder.yaml` |
| Direct addition | `addition.yaml` |
| Cross-attention only | `cross.yaml` |
| Self-attention + addition | `self_addition.yaml` |
| K=1,3,4,5 (K=2 is the base config) | `k1.yaml`, `k3.yaml`, `k4.yaml`, `k5.yaml` |

```bash
uv run --frozen python main.py --config config/ablations/shared_encoder.yaml
```

Configs support relative `extends` and recursive mapping overrides; lists replace parent lists. To inspect or adapt the historical baseline implementations, the original development configurations remain in `config/legacy/index-development.yaml`, `config/legacy/index-seed.yaml`, and `config/legacy/index-mled.yaml`. These are legacy research settings, not ECFNet paper presets. Baseline adapters in `net/baseline/` and `net/emt.py` are retained; their training details are not fully specified by the ECFNet manuscript and their existence does not establish reproduction of Table 1. The manuscript's SEED-IV conclusion has no accompanying detailed protocol or result table; no verified SEED-IV reproduction is claimed here.

## Outputs and checkpoint evaluation

Outputs are separated by `training.exp_name`:

```text
run/checkpoints/<exp_name>/   # best and last epoch checkpoints per fold
run/records/<exp_name>/       # resolved config, fold manifests, normalization,
                             # histories, predictions, summaries, figures
run/tensorboard/<exp_name>/   # per-fold TensorBoard records
run/logs/<save_dir>/          # console logs
```

Per-subject accuracy and macro-F1 are averaged with equal subject weighting. Standard deviations use `ddof=0`, inherited from the source; JSON scores use fractions, not percentages. The pooled row-normalized confusion matrix and one-vs-rest class AUCs use restored-model predictions. Each fold records training, validation and test subject IDs, seed, sample counts, checkpoint name and normalization file. Preserve both its checkpoints and records to re-evaluate it.

```bash
uv run --frozen python main.py --config config/index-strict.yaml --evaluate
# A partial run can be re-evaluated with --evaluate --folds 1 2.
```

Evaluation loads the matching checkpoint **for each held-out subject**, applies saved training statistics, and writes to `run/records/<exp_name>/evaluation/`. A single checkpoint cannot be used as a LOSO model for all subjects. Configuration mismatches are rejected.

Optional interpretation uses only the held-out subject with its saved normalization:

```bash
uv sync --frozen --extra visualization
uv run --frozen --extra visualization python visualize.py --config config/index-strict.yaml --subject 1
```
