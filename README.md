
# 1. Introduction 
Ruiyao Chen     Ruiyao.Chen@Vanderbilt.edu

This project aims at developing a machine-learning algorithm to diagnose precipitation types from liquid, mixed and ice phases using active and passive microwave observations.

Specifically speaking, it combines Convolutional Neural Network (CNN) and Resnet18 to extract the high-level features from different dimension space so that the determined parameters can capture the complicated characteristics of the data from different angles.

The dataset was split into 3 sub-datasets: 70% for training, 20% for testing, and 10% for validation.

Each sub-dataset has 13 features and 2 labels. The features include the scattering-induced brightness temperatures depressions (deltaTB) and corresponding observational zenith angles. The differences between the passive microwave observations and the clear-sky radiative transfer simulations were regarded as deltaTBs . The labels are corresponding to the precipitation types from GMI/DPR.
For better classification performance, I used two types of labels. the first type (0/1) only indicates no/rain and rain. The second (0/1/3/4) provides more details for rain events by indicating no rain/liquid/ice/mixed.

# Project and Code Structure

```
├── .run/
├── atmos/
├── docs/
├── examples/
│   ├── config
│   ├── data
│   ├── output
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── run.py
└── setup.py
```
* **.run/**  : This directory holds shell scripts and Pycharm run configurations for training and testing.
* **atmos/**  : The directory contains source code, but now there are subdirectories. 
* **docs/** :  The documents provided by this project
* **examples/** : A list of datasets, and their corresponding YML configurations files. The output of models also will be saved into the corresponding dataset folder. 
* **LICENSE** : The license for this project
* **Makefile** : A makefile to compile this project  
* **requirements.txt** : A list of required python packages
* **run.py** : A run file for python projects  
* **setup.py** : A setup file for python projects

To run the examples, you need the following programs installed:
- **make**
- **conda**
- **cuda**

## Software Package Installation and Configuration

1. Install CUDA and GPU Driver if using GPU (We are using cuda-10.1)

2. Install latest miniconda as Python virtual environment management
   
3. Create a conda environment using a YML config file:
```
conda env create --name "atmosphere" -f atmosphere.yml
```
Or just execute make command to install all needed packages automatically.
```
make
```

4. Activate Python Virtual Environment
```
conda activate atmosphere
```

5. A few other frequently used commands
```
# list all the conda environment available
conda info --envs  
# Create new environment named as `envname`
conda create --name envname
# Remove environment and its dependencies
conda remove --name envname --all
# Clone an existing environment
conda create --name clone_envname --clone envname
```

# Data Description

- Input data format
- Output lable format

# How to run model

Our tool provides three major python script commands for users:
- **python run.py train:** Training models with different parameter settings to approach a stable state and also save checkpoints for future testing. Using the following command to get more helpful information for each parameter.
```
python run.py train --help
```

- **python run.py predict:** Performing models to classify context labels for test data using the best model checkpoint. Using the following command to get more helpful information for each parameter.

```
python run.py predict --help
```


- **python run.py run:** Training models with different parameter settings to approach a stable state, and then predict context labels for test datasets.

```
python run.py run --help
```


There are two ways to specify parameter settings for each command:

- **YML-based configuration file**, You can specify the values of parameters in a YML config file (See more examples in the config folder of each dataset)
```
$ python run.py "train" --config examples/config/resnet_v0.yml
```

- **Command line arguments**, you can append line arguments with different parameters indicators

```
$ python run.py "test" --config examples/config/resnet_v0.yml \
        --device_id 1 \
        --batch_size 16
```