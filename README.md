
# 1. Introduction 
Ruiyao Chen     Ruiyao.Chen@Vanderbilt.edu

This project aims at developing a novel CNN-based deep learning model with capability of attention mechanism to diagnose precipitation types from liquid, mixed and ice phases using active and passive microwave observations. The experimental results show superior performance. 

Once the data preprocessing including collocation and data cleaning, we then split resulted paired matchups into three subsets, training, testing and validation, in a ratio of 8:1:1. Each subset has 13 features and 2 labels. The features include the scattering-induced brightness temperatures depressions (delta_TB) and corresponding observational zenith angles. The differences between the passive microwave observations and the clear-sky radiative transfer simulations were regarded as deltaTBs. The labels are corresponding to the precipitation types from GMI/DPR. 


This project aims at designing an enhanced CNN model with hybrid local and global attention mechanisms for classifying precipitation types.  Particularly, the model was developed atop of 1D ResNet-18 with following modifications:  
1) Strengthened functionality of the core bottleneck block in ResNet-18 by replacing spatial convolution with a local self-attention layer to capture spatial knowledge of an observations among local area. 
2) Inserted a global self-attention module before the last 1x1 downsampling convolution to exploit global relationships among features. All other structures, including the number of layers and spatial downsampling, were preserved. 

In addition, we leverage multi-task learning techniques to optimize the proposed model with respect to two sub-tasks. The first task is a binary classification problem to identify whether an observation is a case of rain or no-rain event; The second one is a multinomial classification issue of classifying instances into four fine-grained classes: no rain, liquid, ice and mixed. 

In the early develop phase, we also implemented the following baseline models, and we included them in this package for reference purposes. 
-	Gaussian Naive Bayes (GaussianNB)
-	Random Forest
-	ResNet-18 with only one task
-	A simple Multilayer Perceptron (MLP) model
-	Self-Attention model (MLP with attention)



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
        --checkpoint 
```