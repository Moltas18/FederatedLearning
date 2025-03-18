'''
This file can be used to generate parameters. Analysis is done in a seperate file.
'''


import sys
import os
import torch

if __name__ == '__main__':
    
    # Add the root directory to the sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Import local modules
    from src.utils import parse_run, deserialize_parameters, read_from_file, get_filenames, dict_list_to_dict_tensor, set_parameters, denormalize
    from src.plots import plot_reconstruction
    from src.attack.utils import ParameterDifference, GradientApproximation
    from data.data import Data
    from src.models.CNNcifar import CNNcifar
    from src.attack.SME import SME
    from src.metrics.metrics import SSIM, LPIPS, PSNR, MSE


    run_path = 'results/2025-03-18/13-46-05/'
    df = parse_run(run_path = run_path)
    run_idx = 0

    ### Basic example usage ###
    run_series = df.iloc[run_idx]

    print(run_series)
    # Load the parameters from the first client, the round.
    initial_params, updated_params = run_series['Initial Parameters'], run_series['Updated Parameters']

    # Load hyperparameters from the first round
    epochs = run_series['Epochs']
    lr = run_series['Learning Rate']
    partition_id = run_series['Partition ID']

    ### Data configurations
    batch_size = 'full'
    val_test_batch_size = 256
    val_size = 0.5 # 50% of the data is used for validation (we use one image for training and one for validation)
    partitioner = int(12500/2)
    seed = 42

    data = Data(batch_size=batch_size,
                partitioner=partitioner,
                seed=seed,
                val_size=val_size,
                val_test_batch_size=val_test_batch_size)
    
    # W0, WT = deserialize_parameters(initial_params), deserialize_parameters(updated_params)
    
    # comperator = ParameterDifference(W0=W0, W1=WT)
    # comperator.plot_difference()

    # Load the image of the selected partition
    trainloader, _, _ = data.load_datasets(partition_id=partition_id)
    # batch = next(iter(trainloader))
    # x = batch['img']
    # y = batch['label']

    # plot_reconstruction(gt_image, recon_image)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global_params = dict_list_to_dict_tensor(initial_params)
    updated_params = dict_list_to_dict_tensor(updated_params)
    
    # Attack parameters
    alpha = 0.5
    mean_std = torch.tensor([0.5, 0.5, 0.5], device=device)
    lamb = 0.01

    net = CNNcifar

    sme = SME(
        trainloader=trainloader,
        net=net,
        w0=global_params, 
        wT=updated_params,
        device=device,
        alpha=alpha,
        mean_std=mean_std,
        lamb=lamb
    )

    eta = 1
    beta = 0.001
    iters = 1000
    lr_decay = False
    predicted_images, true_images, true_labels =  sme.reconstruction(eta,
                                                                    beta,
                                                                    iters,
                                                                    lr_decay)
    
    # Detach, clone, and denormalize images, this should probably be done outside!
    ground_truth_images = denormalize(true_images.clone().detach())
    reconstructed_images = denormalize(predicted_images.clone().detach())
    
    # Calculate SSIM score between ground truth and predicted images
    ssim_value = SSIM(reconstructed_images, ground_truth_images)
    print(f"SSIM between reconstructed and ground truth images: {ssim_value}")
    
    # Calculate LPIPS score between ground truth and predicted images
    lpips_value = LPIPS(predicted_images, true_images)
    print(f"LPIPS score between predicted and ground truth images: {lpips_value}")

    # Calculate PSNR score between ground truth and predicted images
    psnr_value = PSNR(predicted_images, true_images)
    print(f"PSNR score between predicted and ground truth images: {psnr_value}")

    # Calculate MSE score between ground truth and predicted images
    mse_value = MSE(predicted_images, true_images)
    print(f"MSE score between predicted and ground truth images: {mse_value}")


    
    plot_reconstruction(ground_truth_images=ground_truth_images, reconstructed_images=reconstructed_images)