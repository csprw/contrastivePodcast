# Self-supervised Contrastive Learning from Podcast Audio and Transcripts
<!-- ### Podcast representation learning -->
<!-- <p align="center"> -->
<br><br>
<!-- <img src="./figures/Acrobot-v1_pair_plot.png"
     alt="A paired-plot for metrics in the Acrobot-v1 environment"
     style="float: center;" />
</p> -->


## General info 

This repository contains a framework to learn multi-modal podcast representations, based on podcast raw audio files and transcripts. Using a contrastive loss function we optimize to learn a latent space in which similar transcript and sentences are close together, while pushing dissimilar samples further apart. 


## Folders included

- src                        : Python files to train the framework, logs, and results. 
- data                       : Folder containing the Spotify Podcast Dataset.
- figures                    : Figures.


## Usage 
```
python train.py --args
```

## Getting started 
### 1a. Set up your own python environment.
> The following package versions are used:
- Python 3.9.7
- PyTorch 1.11.0
- pandas 1.3.4
- numpy 1.20.0
- transformers 4.16.2
- h5py 3.6.0

### 1b. Or clone our conda environment.
> Create conda environment from .yml:
```
conda env create -f environment.yml
```

### 2. Acquire data
1. Request access to the SP dataset
2. Place the dataset files in ./data/

### 3. Preprocess data
1. Follow all steps in src/scripts/ to start preprocess all data. 

### 4. Run the training fild:
```
python train.py --args
```
