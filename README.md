# Pretraining on Dynamic Graph Neural Networks

Our article is [PT-DGNN](https://arxiv.org/abs/2102.12380) and the code is modified based on [GPT-GNN](https://github.com/acbull/GPT-GNN)

## Requirements

* python 3.6
* Ubuntu 18.04.5 LTS
* torch 1.4.0+cu100
* more packages in requirements.txt

## Runing example

1. pip install -r requirements.txt
2. python pretrain.py --args values(args and values are detailed instructions in the code)
3. python finetune.py --args values
