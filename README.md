# Federated Learning Thesis  
This repository contains the code for the **Federated Learning Master's Thesis** by Malte Olsson & Carl Kronqvist.  

## Table of Contents  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
- [Running Experiments](#running-experiments)  
- [GPU Acceleration](#gpu-acceleration)  
- [License](#license)  

## Installation  
Python 3.12.9 has been used during development of this repository. It is unclear if the environment can be set up with any other version.

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

### GPU Acceleration (Optional)  
For improved performance, CUDA is recommended if GPUs are available.  
- This project was developed using **CUDA 12.6** on **Windows 10**, but compatibility with other versions has not been tested.  

## Project Structure  
```plaintext  
federated-learning-thesis/  
├── data/                       # Holds datasets and the Data class that handles partitioning
├── experiments/                # Experiment scripts  
├── src/                        # Source code  
│   ├── client/                 # Client-side federated learning code  
│   ├── server/                 # Server-side federated learning code  
│   ├── utils/                  # Utility functions for data handling, evaluation, etc.  
│   └── __init__.py             
├── notebooks/                  # Jupyter notebooks  
├── logs/                       # Logs for experiments 
├── results/                    # Experiment results
└── tests/                      # Unit tests and integration tests  
    ├── test_client.py          # Tests for client code  
    ├── test_server.py          # Tests for server code  
    └── __init__.py  
```  

## Getting Started  
To get familiar with the repository structure, try running the simple federated learning example, `experiments/tutorial.py` or equivilantly `notebooks/tutorial.ipynb`
which is an adaption of the the [Get started with Flower](https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html)-tutorial.  

### Running a Basic Example  
Run the following command to start a basic federated learning setup:  
```sh  
python experiments/template/template.py  
```  
The results and corresponding configurations for the specific run will be saved in `results/`