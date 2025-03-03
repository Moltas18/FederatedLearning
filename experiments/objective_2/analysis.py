'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''

import sys
import os
import pandas as pd
from typing import List, Tuple, Union
from pathlib import Path
if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, deserialize_parameters, read_from_file, get_filenames
    from src.attack.utils import ParameterDifference, GradientApproximation

    run_path = 'results/2025-03-03/09-05-30/'
    df = parse_run(run_path = run_path)



    ### Basic example usage ###

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

    grad_approximator = GradientApproximation(W0=W0,
                                              W1=W1,
                                              lr=lr,
                                              epochs=epochs,)
    approximated_gradients = grad_approximator.approximate_gradient()   

    

