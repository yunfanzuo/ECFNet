# SEED data and preprocessing

Obtain the original SEED **200-Hz, 62-channel preprocessed EEG recordings** through the dataset provider, then place them as follows (or change `dataset.raw_root`):

```text
data/raw/SEED/Preprocessed_EEG/
  label.mat
  1_<session-date>.mat    # three recording files per subject
  ...
  15_<session-date>.mat
```

Each recording must contain trials `<prefix>_eeg1` through `<prefix>_eeg15`. Trials are sorted by numeric suffix and labeled using `label.mat`; labels −1/0/+1 map to negative/neutral/positive indices 0/1/2. Session files are sorted by date. All 15 subjects and three sessions are used by the supplied configs. The program reports the actual generated sample count; the paper reports 35,190 segments from 675 trials. This count cannot be checked without the recordings.

Preprocessing uses 20-s segments at a 4-s stride and 37 windows per segment (2 s, stride 0.5 s). DE uses fourth-order Butterworth bands 1–4, 4–8, 8–14, 14–31, and 31–50 Hz and population variance. Descriptors use a separate 1–64 Hz signal and four-level db4 DWT with symmetric extension. **Wavelet scales are distinct from the DE bands.** Zero total wavelet energy, nonpositive logarithm inputs, and nonfinite features raise errors rather than silently producing invalid training inputs.

The cache contains `data.h5`, `index.parquet`, and `config.json` in `data/processed/SEED/<config_name>/`. A preprocessing fingerprint invalidates caches when settings change. After changing raw recordings at the same path, use `--rebuild-data`. Rebuilding replaces that configuration's feature file. The strict and legacy evaluation configs can share raw feature caches because their normalization happens after loading.
