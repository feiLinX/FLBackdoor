# Introduction
These are the source codes of the implementation our cross-domain FL backdoor attack leveraging feature non-IIDness and the proposed Gradient-inversion Reconstruction-based Anomaly Detection (GRAD) defense.

# Environment
Experiments were run with the following software versions:

| Package | Version |
| --- | --- |
| Python | 3.10.16 |
| PyTorch (`torch`) | 2.7.0 (CUDA 12.8 build) |
| `torchvision` | 0.22.0 |
| `torchaudio` | 2.7.0 |
| NumPy | 2.0.1 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| Pillow | 11.0.0 |
| Matplotlib | 3.10.0 |

Built against CUDA 12.8 and cuDNN 9.7.1.

# Data
Digits-5: https://drive.google.com/file/d/1RemE1_6K-laAN-yHtSb8uDhrgBdXosIg/view?usp=sharing
DomainNet: https://drive.google.com/file/d/1ybeC4FlmL3CA3brEhiW9CkOPaakJNkzQ/view?usp=sharing

# How to run
- `main.py`: `--data_dir`, `--logdir`, and `--ckptdir` contain no hardcoded absolute paths. Their defaults are computed relative to the location of `main.py` (via `os.path.dirname(__file__)`), resolving to `<repo>/../../data/`, `<repo>/logs/`, and `<repo>/saved_models/` respectively. Pass the corresponding flag on the command line if your data lives elsewhere.
- Run scripts under `scripts/`: the interpreter is selected via `PYTHON="${PYTHON:-python}"`, i.e. the `python` found on your `PATH` (e.g. an activated conda environment) is used by default. Set `export PYTHON=/path/to/python` to point at a specific interpreter.
- An example: to test the Full attack against FLAME on Digits-5 with MobileNetV2 as the model, run `./digits_mobilenetv2_nclients_20_agg_flame_att_pred+model_full.sh` in `scripts/`.

