# Code for "Improving Online Continual Learning Performance and Stability With Temporal Ensemble"

Code for the paper [Improving Online Continual Learning Performance and Stability With Temporal Ensemble](https://arxiv.org/abs/2306.16817), Soutif--Cormerais et. al., CoLLAs 2023.

The code is based on [Pytorch](https://pytorch.org) and [Avalanche](https://avalanche.continualai.org).

# Installation

Install pytorch with conda ([Instructions](https://pytorch.org))

install the environment and update it using the environment file

```
cd avalanchev3
conda env create -f environment.yml
conda env config vars set PYTHONPATH=online_ema.git_path:avalanchev3_path
```

Change the data directory DATADIR inside toolkit/dataset.py to match the one on your system

Create the results dir

# How to use

To run single experiments, find the appropriate config file in config dir an run

```
python main_noboundaries.py --config config/experiment_config.yml
```

Add the EMA ensembling and parallel evaluation for more efficient continual evaluation

```
python main_noboundaries.py --config config/experiment_config.yml --mean_evaluation --parallel_evaluation --eval_every 1
```

# License

Please check the License file listed in this repository.

# Cite

```
@inproceedings{soutifcormerais2023improving,
  title={Improving Online Continual Learning Performance and Stability with Temporal Ensembles},
  author={Soutif--Cormerais, Albin and Carta, Antonio and Van de Weijer, Joost},
  booktitle={Conference on Lifelong Learning Agents},
  pages={828--845},
  year={2023},
  organization={PMLR}
}
```
