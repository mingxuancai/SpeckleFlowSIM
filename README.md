# Dynamic Structured Illumination Microscopy with a Neural Space-time Model

## [Project Page](https://rmcao.net/project/speckleflowsim/) | [Paper](https://arxiv.org/abs/2206.01397) | [Experimental Data](https://drive.google.com/file/d/19iE_iUenZdXmnuAIX6lodqG-NRBrf-p4/view?usp=sharing)

## Prerequisite
- [CUDA 11.X](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (JAX distribution is CUDA version-specific)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- [Anaconda](https://www.anaconda.com/products/individual)

## Setup
Clone this project

```
git clone https://github.com/Waller-Lab/SpeckleFlowSIM.git
```

Set up & activate virtual env
```
conda create -n virtualenv_name python=3.9
conda activate virtualenv_name
```

Install dependencies
```
pip install https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.72+cuda111-cp39-none-manylinux2010_x86_64.whl
pip install -r requirements.txt  # install the rest of env via pip
conda install -c conda-forge jupyterlab nodejs ipympl  # for visualization
```
Install the in-house library
```
git clone --branch v0.0.1 https://github.com/rmcao/CalCIL.git
cd calcil
pip3 install -e .
```

## Download experimental data
[Download the data from Google Drive](https://drive.google.com/file/d/19iE_iUenZdXmnuAIX6lodqG-NRBrf-p4/view?usp=sharing) and place it under the project folder.


## Open Jupyter
```
$ jupyter lab --no-browser --port=8899
```

## Try Speckle Flow SIM on Jupyter notebooks 
[simulation.ipynb](https://github.com/rmcao/SpeckleFlowSIM/blob/main/simulation.ipynb): simulation reconstruction on a dynamic Shepp-Logan phantom.

[experiment.ipynb](https://github.com/rmcao/SpeckleFlowSIM/blob/main/experiment.ipynb): experimental reconstruction on a absorptive USAF-1951 resolution target.


## Folder structure
```
├── checkpoint          : folder to store model checkpoints
├── README.md           : README file
├── simulation.ipynb    : notebook for Speckle Flow SIM simulation
├── experiment.ipynb    : notebook for Speckle Flow SIM experiment
├── experiment.npz      : experimental data
├── requirement.txt     : dependencies to install
├── spacetime.py        : implementation of the neural space-time model
├── speckle_flow.py     : incorporating Speckle SIM forward model with neural space-time model for Speckle Flow SIM
└── utils.py            : utility functions for motion and dynamic scene generation.
```

## Citation
```
@article{cao2022dynamic,
  title={Dynamic Structured Illumination Microscopy with a Neural Space-time Model},
  author={Cao, Ruiming and Liu, Fanglin Linda and Yeh, Li-Hao and Waller, Laura},
  journal={arXiv preprint arXiv:2206.01397},
  year={2022}
}
```
