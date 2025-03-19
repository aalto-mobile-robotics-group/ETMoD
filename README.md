# ETMoD

This repository contains code for the paper Event-Triggered Maps of Dynamics: A Framework for Modelling Spatial Motion Patterns in Non-Stationary Environments.

## About

This work automatically identifies regions of interest and uses Neural SDEs to model temporal changes in motion patterns. A diffusion model is integrated to generate intermediate data for the framework. 
<div align='center'>
  <br><img src="ETMoD.gif" width=70%>
</div>

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

### What can I find in this repository?
The repository is accompanied by scripts showing how to use ETMoD:
* *gridshift.py* - This example shows how to cluster the spatial data out of a set of observations collected in a set of locations.

* *gs_em.py* - This example shows how to obtain the multi-modal distribution of the velocity data in each context-awared cell.

* *sde_diff.py* - This script contains our SDE model for temporal modelling.

* *diffusion.py* - This script contains our diffusion model for generating intermediate data.

### Data
The testing data is from the [ATC shopping center tracking dataset](https://dil.atr.jp/crest2010_HRI/ATC_dataset/).

## Acknowlegement
Part of our code is reimplemented from [CLiFF-Map](https://github.com/tkucner/CLiFF-map-matlab). We thank the authors for releasing their codes.
