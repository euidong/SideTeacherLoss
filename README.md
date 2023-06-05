# SideTeacherLoss

<div align="center">
  
  ![python](https://img.shields.io/badge/python-3.8.16-brightgreen)
  ![pytorch](https://img.shields.io/badge/pytorch-2.0.1-orange)
  ![torchvision](https://img.shields.io/badge/torchvision-0.15.2-blueviolet)
  ![python](https://img.shields.io/badge/matplotlib-3.7.1-blue)
  
</div>

<div align="center">
  <img src="https://github.com/euidong/SideTeacherLoss/assets/48043626/d6083574-1545-40e4-8ab5-b070a38eb419" width="500px" />
</div>

In AI(ML, DL), Teacher-Student mechanism always transfer good knowledge to student from teacher.
But, we tried knowledge distilation from bad performance teacher(Side Teacher) and use this knowledge for learning of students.
And we can show that Side Teacher can be used for getting better performance student.

## Demo

you can show our demo and result in `demo.ipynb`.
This demo use CIFAR100 dataset.

## Run

### 1. Make sure you installed dependencies.

```bash
$ conda create -n side-teacher python=3.8.16
$ conda activate side-teacher
$ conda install torchvision torch -c torch
$ conda install matplotlib -c conda-forge
```

### 2. Run main

This program has some arguments, you can check it in `main.py` file.

```bash
$ python main.py
```

### 3. Draw Graph

```bash
$ python drawer.py
```

We also have some advanced version of this code in `weighted-alpha`, `max_dist` branch, you can check it.

## Methods

First, we declared side teacher as overfitted model. 
And we tried student's parameter to be far from side teacher's parameter.
So, we add a regularization term(distance from side teacher's parameter) to student's loss function.

$$
\mathcal{L} = f(\theta) - \alpha d(\theta_{student}, \theta_{teacher})
$$

<div align="center">or</div>

$$
\mathcal{L} = f(\theta) + \frac{\alpha}{d(\theta_{student}, \theta_{teacher})}
$$

So, we get below student losses

1. Cross Entropy - $\alpha$ L1-norm
2. Cross Entropy - $\alpha$ L2-norm
3. Cross Entropy - $\alpha$ Frobenius-norm
4. Cross Entropy + $\alpha$ $\frac{1}{L1-norm}$
5. Cross Entropy + $\alpha$ $\frac{1}{L2-norm}$
6. Cross Entropy + $\alpha$ $\frac{1}{Frobenius-norm}$

and compare this model with

1. Cross Entropy
2. Cross Entropy + $\alpha$ weight decay

## Evaluation

### Environment
For evaluation we setup below

1. Same initialization parameter(students, baselines)
2. \# teacher = 10
3. split dataset for overfitting teacher model

<div align="center">
  <img src="https://github.com/euidong/SideTeacherLoss/assets/48043626/91d27311-bdd2-4101-904d-2585cccb54cc"  width="500px" />
</div>

### Result

In many classification dataset, our method can't get noticeable result. 
But, in CIFAR100, we can get better performance compared to baselines(Pure Cross Entropy, Weight Decay + Cross Entropy) 

<div align="center">
  <img src="https://github.com/euidong/SideTeacherLoss/assets/48043626/a6af170f-b89c-4193-88cb-da6b1d88cd7a"  width="800px" />
</div>
