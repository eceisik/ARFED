# BARFED: Byzantine Attack Resistant Federated Averaging Based on Outlier Elimination

This repository is the official implementation of BARFED: Byzantine Attack Resistant Federated Averaging Based on Outlier Elimination. 
The framework and full methodology are detailed in [our manuscript](https://arxiv.org/abs/2111.04550).

## Citation
Our software can be cited as:
```
@misc{isikpolat2021barfed,
      title={BARFED: Byzantine Attack-Resistant Federated Averaging Based on Outlier Elimination}, 
      author={Ece Isik-Polat and Gorkem Polat and Altan Kocyigit},
      year={2021},
      eprint={2111.04550},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Requirements

To install requirements:

```
pip install -r requirements.txt
```
OR    

To create a conda virtual environment from an environment.yml file. Please see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more details.
```
conda env create -f environment.yml
```

## How to run

We performed the experiments for label flipping and Byzantine attacks on MNIST, Fashion MNIST, and CIFAR10 datasets.  We compared our results with the median method. The file name list for each data set is as follows. The default setting of experiments is NON-IID organized with a 20% malicious participant ratio.

| Dataset        | Attack Type        |    Method       | File Name             | 
| ---------------| ------------------ |---------------- | --------------------- |      
| MNIST          | Byzantine          | ARFED           | mnist_byz_barfed.py   | 
| MNIST          | Byzantine          | Median          | mnist_byz_median.py   | 
| MNIST          | Byzantine          | No Defense      | mnist_byz_attack.py   | 
| MNIST          | Label Flipping     | ARFED           | mnist_lf_barfed.py    | 
| MNIST          | Label Flipping     | Median          | mnist_lf_median.py    | 
| MNIST          | Label Flipping     | No Defense      | mnist_byz_attack.py   | 
| Fashion MNIST  | Byzantine          | ARFED           | fashion_byz_barfed.py | 
| Fashion MNIST  | Byzantine          | Median          | fashion_byz_median.py | 
| Fashion MNIST  | Byzantine          | No Defense      | fashion_byz_attack.py | 
| Fashion MNIST  | Label Flipping     | ARFED           | fashion_lf_barfed.py  | 
| Fashion MNIST  | Label Flipping     | Median          | fashion_lf_median.py  | 
| Fashion MNIST  | Label Flipping     | No Defense      | fashion_lf_attack.py  |
| CIFAR10        | Byzantine          | ARFED           | cifar_byz_barfed.py   | 
| CIFAR10        | Byzantine          | Median          | cifar_byz_median.py   | 
| CIFAR10        | Byzantine          | No Defense      | cifar_byz_attack.py   | 
| CIFAR10        | Label Flipping     | ARFED           | cifar_lf_barfed.py    | 
| CIFAR10        | Label Flipping     | Median          | cifar_lf_median.py    | 
| CIFAR10        | Label Flipping     | No Defense      | cifar_lf_attack.py    | 
