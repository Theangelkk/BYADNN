#!/usr/bin/env bash

conda create -n BYADNN_env python=3.9
conda activate BYADNN_env
pip install ipywidgets
conda install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install pyro-ppl==1.8.4
pip install -U scikit-learn
pip install graphviz
pip install omegaconf
conda install python-graphviz -y