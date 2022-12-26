# ARFED: Attack Resistant Federated Averaging Based on Outlier Elimination

This repository is the official implementation of __*ARFED: Attack Resistant Federated Averaging Based on Outlier Elimination*__. 
The framework and full methodology are detailed in [our manuscript](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22004083).

## Install required packages

```
pip install -r requirements.txt
```
for pip environment. OR

```
conda env create -f environment.yml
```

To create a conda virtual environment from an environment.yml file. Please see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more details.


## How to run

We performed the experiments for label flipping and Byzantine attacks on MNIST, Fashion MNIST, and CIFAR10 datasets.  We compared our results with the median method. The file name list for each data set is as follows. The default setting of experiments is NON-IID organized with a 20% malicious participant ratio.

| Dataset        | Attack Type        |    Method       | File Name             | 
| ---------------| ------------------ |---------------- | --------------------- |      
| MNIST          | Byzantine          | ARFED           | mnist_byz_barfed.py   | 
| MNIST          | Byzantine          | Median          | mnist_byz_median.py   | 
| MNIST          | Byzantine          | Trimmed Mean    | mnist_byz_trimmed.py  |
| MNIST          | Byzantine          | No Defense      | mnist_byz_attack.py   | 
| MNIST          | Label Flipping     | ARFED           | mnist_lf_barfed.py    | 
| MNIST          | Label Flipping     | Median          | mnist_lf_median.py    | 
| MNIST          | Label Flipping     | Trimmed Mean    | mnist_lf_trimmed.py   |
| MNIST          | Label Flipping     | No Defense      | mnist_byz_attack.py   | 
| MNIST          | Partial Knowledge  | ARFED           | mnist_partial_knowledge_barfed.py    | 
| MNIST          | Partial Knowledge  | Median          | mnist_partial_knowledge_median.py    | 
| MNIST          | Partial Knowledge  | Trimmed Mean    | mnist_partial_knowledge_trimmed.py   |
| MNIST          | Partial Knowledge  | No Defense      | mnist_partial_knowledge_attack.py    |
| Fashion MNIST  | Byzantine          | ARFED           | fashion_byz_barfed.py | 
| Fashion MNIST  | Byzantine          | Median          | fashion_byz_median.py | 
| Fashion MNIST  | Byzantine          | Trimmed Mean    | fashion_byz_trimmed.py| 
| Fashion MNIST  | Byzantine          | No Defense      | fashion_byz_attack.py | 
| Fashion MNIST  | Label Flipping     | ARFED           | fashion_lf_barfed.py  | 
| Fashion MNIST  | Label Flipping     | Median          | fashion_lf_median.py  | 
| Fashion MNIST  | Label Flipping     | Trimmed Mean    | fashion_lf_trimmed.py |
| Fashion MNIST  | Label Flipping     | No Defense      | fashion_lf_attack.py  |
| Fashion MNIST  | Partial Knowledge  | ARFED           | fashion_partial_knowledge_barfed.py  | 
| Fashion MNIST  | Partial Knowledge  | Median          | fashion_partial_knowledge_median.py  | 
| Fashion MNIST  | Partial Knowledge  | Trimmed Mean    | fashion_partial_knowledge_trimmed.py |
| Fashion MNIST  | Partial Knowledge  | No Defense      | fashion_partial_knowledge_attack.py  |
| CIFAR10        | Byzantine          | ARFED           | cifar_byz_barfed.py   | 
| CIFAR10        | Byzantine          | Median          | cifar_byz_median.py   | 
| CIFAR10        | Byzantine          | Trimmed Mean    | cifar_byz_trimmed.py  | 
| CIFAR10        | Byzantine          | No Defense      | cifar_byz_attack.py   | 
| CIFAR10        | Label Flipping     | ARFED           | cifar_lf_barfed.py    | 
| CIFAR10        | Label Flipping     | Median          | cifar_lf_median.py    | 
| CIFAR10        | Label Flipping     | Trimmed Mean    | cifar_lf_trimmed.py   | 
| CIFAR10        | Label Flipping     | No Defense      | cifar_lf_attack.py    | 
| CIFAR10        | Partial Knowledge  | ARFED           | cifar_partial_knowledge_barfed.py    | 
| CIFAR10        | Partial Knowledge  | Median          | cifar_partial_knowledge_median.py    | 
| CIFAR10        | Partial Knowledge  | Trimmed Mean    | cifar_partial_knowledge_trimmed.py   | 
| CIFAR10        | Partial Knowledge  | No Defense      | cifar_partial_knowledge_attack.py    | 

## Citation
Please use the following BibTeX entry for citation:
```BibTeX
@article{ISIKPOLAT2023626,
title = {ARFED: Attack-Resistant Federated averaging based on outlier elimination},
journal = {Future Generation Computer Systems},
volume = {141},
pages = {626-650},
year = {2023},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2022.12.003},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X22004083},
author = {Ece Isik-Polat and Gorkem Polat and Altan Kocyigit}
}
```
