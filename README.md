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
To install all required dependencies, run:  
```sh  
pip install -r requirements.txt  
```  
If you encounter issues, try:  
```sh  
pip install --no-cache-dir -r requirements.txt  
```  

### GPU Acceleration (Optional)  
For improved performance, CUDA is recommended if GPUs are available.  
- This project was developed using **CUDA 12.8** on **Windows 10**, but compatibility with other versions has not been tested.  
- Download CUDA here: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads).  

## Project Structure  
```plaintext  
federated-learning-thesis/  
├── data/                       # For datasets (consider using symlinks for large datasets)  
├── experiments/                # Experiment scripts and configurations  
├── src/                        # Source code  
│   ├── client/                 # Client-side federated learning code  
│   ├── server/                 # Server-side federated learning code  
│   ├── utils/                  # Utility functions for data handling, evaluation, etc.  
│   └── __init__.py             
├── notebooks/                  # Jupyter notebooks  
├── logs/                       # Logs for experiments 
├── results/                    # Experiment results
├── config/                     # Config files (JSON, YAML, etc.) for experiments and model settings  
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
python experiments/run_example.py  
```  
Modify `config/` files to adjust settings such as the number of clients, training rounds, and hyperparameters.  

## Running Experiments  
1. Configure your experiment settings in the `config/` directory.  
2. Start the federated learning process:  
   ```sh  
   python experiments/train.py  
   ```  
3. Logs will be saved in the `logs/` directory.  
4. Results will be stored in the `results/` directory.  

## License  
This project is for academic use. Licensing terms will be updated later.  

