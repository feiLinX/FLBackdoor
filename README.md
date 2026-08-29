# Introduction
These are the source codes of the implementation our cross-domain FL backdoor attack leveraging feature non-IIDness and the proposed GRAD defense.

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
- Digits-5: https://drive.google.com/file/d/1RemE1_6K-laAN-yHtSb8uDhrgBdXosIg/view?usp=sharing
- DomainNet: https://drive.google.com/file/d/1ybeC4FlmL3CA3brEhiW9CkOPaakJNkzQ/view?usp=sharing
- Before running any script, create a `<repo>/data` and unzip each dataset into it. `<repo>/data` should just be in the same directory as `<repo>/aggregations`, `<repo>/attacks`, etc.

### Expected directory structure
After unzipping, the folders should look as follows (verify these paths so the loaders can find the data).

Digits-5 unzips to `<repo>/data/digits5/`. Each domain folder holds the raw images, and the per-domain `*_train.pkl` / `*_test.pkl` index files sit at the top level:
```
data/digits5/
├── mnist/
│   ├── train_images/
│   └── test_images/
├── mnist_m/
│   ├── train_images/
│   └── test_images/
├── svhn/
│   ├── train_images/
│   └── test_images/
├── syn/
│   ├── train_images/
│   └── test_images/
├── usps/
│   ├── train_images/
│   └── test_images/
├── mnist_train.pkl
├── mnist_test.pkl
├── mnist_m_train.pkl
├── mnist_m_test.pkl
├── svhn_train.pkl
├── svhn_test.pkl
├── syn_train.pkl
├── syn_test.pkl
├── usps_train.pkl
└── usps_test.pkl
```

DomainNet unzips to `<repo>/data/DomainNet/`. Each domain folder contains one subfolder per class, and the `*_train.pkl` / `*_test.pkl` index files live under `pkls/`:
```
data/DomainNet/
├── clipart/
│   ├── bird/
│   ├── feather/
│   └── ...            # one folder per class
├── infograph/
├── painting/
├── quickdraw/
├── real/
├── sketch/
├── pkls/
│   ├── clipart_train.pkl
│   ├── clipart_test.pkl
│   └── ...            # <domain>_train.pkl / <domain>_test.pkl per domain
├── splits/
│   ├── clipart_train.txt
│   └── ...            # <domain>_train.txt / <domain>_test.txt per domain
└── splits_mini/
```

# Repository structure
```
codes/
├── main.py           # entry point: FL training loop + attack/defense dispatch
├── data_utils.py     # data partitioning and dataloaders
├── data_aug_utils.py # augmentation utilities (AutoAugment, etc.)
├── utils.py          # training / evaluation helpers
├── aggregations/     # server rules: fedavg, krum, flame, ndc, deepsight, foolsgold, bnguard
├── defenses/         # grad (proposed GRAD defense), indicator
├── attacks/          # cdls (proposed attack), pgd, neurotoxin, vanilla, chameleon, soda
├── datasets/         # Dataset classes for Digits-5 / DomainNet / Office
├── models/           # ResNet, MobileNetV2, FangCNN
├── scripts/          # ready-to-run experiment configs (see How to run)
└── data/             # datasets you download (see Data)
```

# How to run
- `main.py`: `--data_dir`, `--logdir`, and `--ckptdir` contain no hardcoded absolute paths. Their defaults are computed relative to the location of `main.py` (via `os.path.dirname(__file__)`), resolving to `<repo>/logs/` and `<repo>/saved_models/` respectively. Pass the corresponding flag on the command line if your data lives elsewhere.
- Run scripts under `scripts/`: the interpreter is selected via `PYTHON="${PYTHON:-python}"`, i.e. the `python` found on your `PATH` (e.g. an activated conda environment) is used by default. Set `export PYTHON=/path/to/python` to point at a specific interpreter.
- An example: to test the Full attack against FLAME on Digits-5 with MobileNetV2 as the model, run `./digits_mobilenetv2_nclients_20_agg_flame_att_pred+model_full.sh` in `scripts/`.
- Per-round metrics (Train Acc / Test Acc / ASR) are logged to `<repo>/logs/`; model checkpoints are saved under `<repo>/saved_models/<aggregation>/`.

### Available options (`main.py`)
| Flag | Choices |
| --- | --- |
| `--dataset` | `digits`, `domain`, `cifar10` |
| `--model` | `resnet18`, `resnet34`, `resnet50`, `mobilenetv2` |
| `--aggregation` | `fedavg`, `krum`, `flame`, `ndc`, `grad`, `deepsight`, `foolsgold`, `bnguard`, `indicator` |
| `--adv_type` | `None`, `CDLS`, `PGD`, `Neurotoxin`, `Vanilla`, `Chameleon`, `SoDa` |

CIFAR-10 are downloaded automatically by torchvision; only Digits-5 and DomainNet require the manual download above.

### Script naming convention (`scripts/`)
Each script is named `<dataset>_<model>_nclients_<N>[_<extra>]_agg_<aggregation>_att_<variant>.sh` and just calls `main.py` with the matching flags. The `att_<variant>` token encodes the attack:

| Token | Meaning |
| --- | --- |
| `none` | clean training, no attack (`--adv_type None`) |
| `raw+data` | our attack, raw-pixel feature distance, data poisoning only (`--bd_distance raw_kl`) |
| `embed+data` | our attack, SimCLR Encoder for feature distance, data poisoning only (`--bd_distance embed_kl`) |
| `pred+data` | our attack, Encoder + LP for feature distance, data poisoning only (`--bd_distance pred_kl`) |
| `woSS+data` | our attack without the selection strategy, data poisoning only (`--bd_distance random`) |
| `pred+model` / `pred+model_full` | `pred+data` plus full model poisoning (`--bd_model_poison`: stealth reg + constrain-and-scale) |
| `pred+model_woCS` | model poisoning without constrain-and-scale (`--bd_scale 1.0`) |
| `pred+model_woSR` | model poisoning without the stealth regularizer (`--bd_stealth_lambda 0.0`) |

Optional `<extra>` tokens: `bias_<b>` sets the label non-IID degree (`--bias`); `nbyz_<k>_bdpart_<p>` sets the number of adversarial clients (`--nbyz`) and the poisoning fraction (`--bd_partition`); `grad_B<r>_T<i>` sets the GRAD defense budget (`--def_num_recon <r>`, `--def_recon_iters <i>`).

