# ncssm_lectures

## Quick Start! (Assuming you have `git` installed)

### Install Conda

From <https://docs.anaconda.com/miniconda/install/>:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ./Miniconda3-latest-MacOSX-x86_64.sh
rm Miniconda3-latest-MacOSX-x86_64.sh
```

### Clone the repository

`git clone git@github.com:rgross-utexas/ncssm_lectures.git`

### Build the conda environment

`conda env create -f environment.yaml`

### Activate the conda environment

`conda activate ncssm_lectures`

### Run the code!

#### Q-learning

#### Reinforce

This is ***SUPER*** CPU intensive and, depending on your setup, could take a few hours! :(

`python ./reinforce.py`

This also produces Tensorboard metrics, so start up Tensorboard a terminal that has the Conda environment activated:

`tensorboard --logdir=output/tb`

Now, go to `http://localhost:6006/` in your browser to watch the metrics!

## Useful Links

### Github Repository

<https://github.com/rgross-utexas/ncssm_lectures>

- code
- configuration
- notes

### Python

<https://docs.anaconda.com/miniconda/install/>


### Reinforcement Learning

- The "bible" for Reinforcement Learning: <http://incompleteideas.net/book/the-book-2nd.html>

- Gyms - <https://gymnasium.farama.org/>
  - <https://gymnasium.farama.org/environments/toy_text/cliff_walking/>
  - https://gymnasium.farama.org/environments/box2d/lunar_lander/

## Creating your Python Conda Environment

### Install Conda

<https://docs.anaconda.com/miniconda/install/>


```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash ./Miniconda3-latest-MacOSX-x86_64.sh
rm Miniconda3-latest-MacOSX-x86_64.sh
```

`conda env create -f environment.yaml`