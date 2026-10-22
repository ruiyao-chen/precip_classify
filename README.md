# precip_classify: precipitation type from passive microwave with attention-augmented CNNs

Ruiyao Chen · Ruiyao.Chen@Vanderbilt.edu

A CNN-based deep-learning model with local and global attention that diagnoses
precipitation phase (liquid, mixed, ice) from passive microwave observations.
This is the code behind:

> R. Chen and R. Bennartz, "A Deep-Learning Scheme for Hydrometeor Type Classification
> Using Passive Microwave Observations," *Remote Sensing*, 15(10), 2670, 2023.
> [doi:10.3390/rs15102670](https://doi.org/10.3390/rs15102670)

The published model reaches **84.7% classification accuracy** on held-out test data,
and its global annual cycles of precipitation occurrence agree with the GPM
Dual-frequency Precipitation Radar (DPR).

# 1. Introduction

After preprocessing (collocation and data cleaning), the paired matchups are split
into training, validation and test subsets in an 8:1:1 ratio. Each sample has 15
features and 2 labels. The features are the scattering-induced brightness-temperature
depressions (ΔTB), the observation zenith angles, and the geographic coordinates. ΔTB is
the difference between the passive microwave observation and a clear-sky
radiative-transfer simulation. The labels are precipitation types from GPM GMI/DPR
(Label 1 and Label 2).

The model combines a CNN with hybrid local and global attention. It is built on a 1D
ResNet-18 with two modifications:

1. The core bottleneck block is strengthened by replacing the spatial convolution with
   a local self-attention layer, to capture spatial context around each observation.
2. A global self-attention module is inserted before the last 1×1 down-sampling
   convolution, to exploit global relationships among features. All other structure,
   including the number of layers and spatial down-sampling, is preserved.

The model is trained with **multi-task learning** on two sub-tasks:

- **Label 1:** binary classification, rain vs. no rain.
- **Label 2:** four-class classification: no rain, liquid, ice, mixed.

Baseline models implemented during development are included for reference:

- Gaussian Naive Bayes
- Random forest
- ResNet-18 with a single task
- A simple multilayer perceptron (MLP)
- A self-attention model (MLP with attention)

# 2. Project and code structure

```
├── atmos/              source code
│   ├── model/          classification (CNN, ResNet-1D, LSTM, MLP, RF, self-attention,
│   │                   transformer) and regression models, plus DL/ML runners
│   └── utils/          config parsing, data input, early stopping, logging
├── docs/               Sphinx documentation skeleton
├── examples/
│   ├── config/         YAML configurations, one per model
│   ├── data/           train / validation / test matchups (.mat)
│   └── output/         trained ResNet-18 checkpoint (Git LFS)
├── atmosphere.yml      conda environment (Linux, CUDA 10.1)
├── Makefile            creates the conda environment
├── requirements.txt    core Python packages
├── run.py              entry point
└── setup.py
```

# 3. Installation

Requirements: Linux, an NVIDIA GPU with CUDA 10.1 (PyTorch 1.7; the preprocessing utilities use CuPy),
conda, and [Git LFS](https://git-lfs.com) to download the trained checkpoint
(about 110 MB).

```bash
git lfs install
git clone https://github.com/ruiyao-chen/precip_classify.git
cd precip_classify
make                       # = conda env create --name atmosphere -f atmosphere.yml
conda activate atmosphere
```

# 4. How to run

`run.py` has three commands:

- `python run.py train`: train a model and save checkpoints.
- `python run.py predict`: classify the test data with a trained checkpoint.
- `python run.py run`: train, then predict on the test data.

Add `--help` to any command to see its parameters. Parameters can come from a YAML
file (see `examples/config/`) or the command line:

```bash
python run.py train --config examples/config/resnet_v0.yml
python run.py predict --config examples/config/resnet_v0.yml --checkpoint checkpoint_best_195000_0.8626.pt
```

# 5. Citation

```bibtex
@article{chen2023deep,
  title   = {A Deep-Learning Scheme for Hydrometeor Type Classification Using Passive Microwave Observations},
  author  = {Chen, Ruiyao and Bennartz, Ralf},
  journal = {Remote Sensing},
  volume  = {15},
  number  = {10},
  pages   = {2670},
  year    = {2023},
  doi     = {10.3390/rs15102670}
}
```

# License

MIT. See [LICENSE](LICENSE).
