# ETMoD

This repository contains code for the paper Event-Triggered Maps of Dynamics: A Framework for Modelling Spatial Motion Patterns in Non-Stationary Environments.

## About

This work automatically identifies regions of interest and uses Neural SDEs to model temporal changes in motion patterns. A diffusion model is integrated to generate intermediate data for the framework. 

![image (15)](https://github.com/user-attachments/assets/7877d782-c7a9-41d4-b59f-4630d538435f)

## Setup

### Environment

Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

Create a Conda environment:

```
conda create -n ETMoD python=3.10
```

And activate it:

```
conda activate ETMoD
```

Install required libraries:

```
pip install -r requirements.txt
```

