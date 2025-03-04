# Inter-Task Similarity and Knowledge Transfer

This repository contains the implementation of the method presented in the article "Inter-task Similarity Measure for Heterogeneous Tasks". The method measures inter-task similarity between RL tasks with discrete state and action spaces and transfers Q-values from one task to the other.

## Install

Create a conda environment:

```
conda create -n intersim_env python=3.8.5
```

Clone this repository:

```
git clone https://github.com/saSerrano/intersim.git
```

Install with pip:

```
cd intersim
pip install -e .
```

Test installation:

```
python test/test_import.py
```

## Citation

The [published article](https://link.springer.com/chapter/10.1007/978-3-030-98682-7_4) of this project can be cited with the following bibtex entry:

```
@incollection{serrano2021inter,
  title={Inter-task similarity measure for heterogeneous tasks},
  author={Serrano, Sergio A and Martinez-Carranza, Jose and Sucar, L Enrique},
  booktitle={Robot World Cup},
  pages={40--52},
  year={2021},
  publisher={Springer}
}
```
