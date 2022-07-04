# RL for Goal Recognition
Supplementary material for AAAI2022. This packages include the code to generate the results of our paper, besides the R&G approach. The R&G approach is not included in this package, only the necessary code to run the developed methods of our paper.

## Setting up the environment
This package was tested on **Python 3.6.13** and should work on **Python 3.6** or higher.
To install requirements (works on virtualenv and conda), run:
```zsh
pip install -r requirements.txt
```
## Running experiments
To compute the results for **partial observability** *(0.1,0.3,0.5,0.7,1.0)*, run the following command inside the ```src/``` folder:

```zsh
python experiments.py
```
The results are going to be outputed on ```src/results.txt```.
To run the experiments with noise, run the following command inside the ```src/``` folder:

```zsh
python experiments_noisy.py
```
The results are going to be outputed on ```src/results_noisy.txt```.