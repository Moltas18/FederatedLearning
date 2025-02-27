'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''

import sys
import os
import pandas as pd

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, deserialize_parameters
    from src.attack.utils import ParameterDifference, GradientApproximation
    run_path = 'results/2025-02-27/11-01-42/'

    config_path = run_path + 'run_config.jsonl'
    metrics_path = run_path + 'metrics.jsonl'
    parameters_path = run_path + 'parameters.jsonl'

    df = parse_run(config_path, parameters_path)
    
    # Load the parameters from the first round
    W0 = df.iloc[0]['Initial Parameters']
    W1 = df.iloc[0]['Updated Parameters']

    # Load hyperparameters from the first round
    epochs = df.iloc[0]['Epochs']
    lr = df.iloc[0]['Learning Rate']

    # Deserialize the parameters into tensors
    W0 = deserialize_parameters(parameters_record=W0)
    W1 = deserialize_parameters(parameters_record=W1)

    comparator = ParameterDifference(W0, W1)
    comparator.plot_difference()

    grad_approximator = GradientApproximation(W0,
                                              W1,
                                              lr=0.1,
                                              epochs=1,)
    approximated_gradients = grad_approximator.approximate_gradient()    

    for i, grad in enumerate(approximated_gradients):
        print(f"Approximated gradient for layer {i}: {grad}")