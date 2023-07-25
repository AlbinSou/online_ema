# Code for "Efficient Ensembling in continual learning"

Code for the paper [Efficient Ensembling in continual learning](https://arxiv.org/abs/2306.16817), Soutif--Cormerais et. al., CoLLAs 2023.

The code is based on [Pytorch](https://pytorch.org) and [Avalanche](https://avalanche.continualai.org).

# Installation

Install pytorch with conda ([Instructions](https://pytorch.org))

install the environment and update it using the environment file

```
cd avalanchev3
conda env create -f environment.yaml
```

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
@misc{soutifcormerais2023improving,
      title={Improving Online Continual Learning Performance and Stability with Temporal Ensembles}, 
      author={Albin Soutif--Cormerais and Antonio Carta and Joost Van de Weijer},
      year={2023},
      eprint={2306.16817},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
