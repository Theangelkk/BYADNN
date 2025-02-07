# Predicting ground-level nitrogen dioxide concentrations using the BaYesian Attention-based Deep Neural Network (BYADNN)

This repository is an offical implementation of the paper "Predicting ground-level nitrogen dioxide concentrations using the BaYesian Attention-based Deep Neural Network" to be published on Ecological Informatics.

## Installation
This code is based on python 3.9, CUDA version 11.3 and pytorch 1.12.1. Additional python packages required to run this repository are: pyro 1.8.4 and scikit-learn and omegaconf. 

All required packages are automatically installed by running the provided `install.sh` with the following command:

```bash
./install.sh
```

Note that, [conda](https://docs.anaconda.com/miniconda/install/) package is required to install this repository.

## Dataset preparation
The Italian $NO_2$ European Environment Agency (EEA) stations of year 2020 data set can be downloaded from [Google Drive directory](https://drive.google.com/drive/folders/19maTuVV-N3zyHgjGrluiEwzK-yKwXCs4?usp=sharing). By default, the provided dataset must be placed in the same directory of the code.

## Execution
This repository can be used by runnign the notebook `main.ipynb`. It is important to set the `path_dir_notebook_file` variable, defining the global path of the repository.

### Training
To perform the code for training phase, the notebook `main.ipynb` can be used, setting the following variables:
* to start a training from scratch:
    * `dataloader_saved = False`;
    * `resume_training = False`;
    * `eval_model = False`;

* to resume a previous training:
    * `dataloader_saved = True`;
    * `resume_training = True`;
    * `eval_model = False`.

### Inference
After training phase, the same notebook file can be run for inference phase by setting the following variables:
* inference:
    * `dataloader_saved = True`;
    * `resume_training = False`;
    * `eval_model = True`.

## Citing

If you use this repository for your research, please consider citing the corresponding paper:

```latex
T.B.D.
```

