# Federated Learning Thesis  
This repository contains the code for the **Federated Learning Master's Thesis** by Malte Olsson & Carl Kronqvist.  

## Table of Contents  
- [Installation](#installation)  
- [GPU Acceleration](#gpu-acceleration)  
- [Project Structure](#project-structure)   
- [Running Experiments](#running-experiments)   

## Installation  
Python 3.12.9 has been used during development of this repository. We recommend using this version as we cannot guarantee compatability with other distributions.

We strongly recommend using a virtual environment for dependencies. To set it up, run:   
```sh  
python -m venv env
```

Before installing any dependencies, make sure to activate it by running: 

```sh  
.\env\Scripts\activate
```

Once the environment is activated - to install all required dependencies, run:  
```sh  
pip install -r requirements.txt  
```  
If you encounter issues, try:  
```sh  
pip install --no-cache-dir -r requirements.txt  
```  

## GPU Acceleration
For improved performance, CUDA is recommended if GPUs are available.  
- This project was developed using **CUDA 12.6** on **Windows 10**, but compatibility with other versions has not been tested.  

## Project Structure  
```plaintext  
federated-learning-thesis/  
├── data/                       # Holds datasets and the Data class that handles partitioning
├── experiments/                # Experiment scripts where main code is run  
├── src/                        # Source code
│   ├── attack/                 # Code for attack methods
│   ├── metrics/                # Code for metrics
│   ├── models/                 # The ML-models used
│   ├── client_app.py           # Client-side federated learning code  
│   ├── plots.py                # Plots
│   ├── simulation.py           # Holds the Simulation class which runs federated learning
│   ├── strategy.py             # Server-side federated learning code  
│   ├── utils.py                # Utility functions
│   └── __init__.py             # Module initializer
├── notebooks/                  # Jupyter notebooks, used mainly for displaying results  
└── results/                    # Experiment results
```  

## Running Experiments  
The source code in `src/` is run through `main.py` files in  `experiments/`.
- In `experiments/objective_1/main.py` & `experiments/objective_2/main.py`, federated learning is run with the option to store parameters throughout the training cycle. If opted to save, parameteres will be stored in .jsonl-files in `results/` according to their time stamp.
- In `experiments/objective_3/main.py`, a deep leakage attack can be carried out on the stored parameters.
- Finally, with `experiments/objective_4/main.py`, restored images from the deep leakage attack can be evaluated.