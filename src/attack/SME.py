import torch
import torchvision
from tqdm import tqdm
import os
import copy
from src.attack.utils import require_grad, prior_boundary, compute_norm, total_variation, save_args, save_figs, psnr
from src.utils import set_parameters
from PIL import Image
import numpy as np

class SME:

    def __init__(
            self,
            trainloader,
            net: torch.nn.Module,
            w0, 
            wT,
            device,
            alpha,
            mean_std,
            lamb,
            dataset=None,
    ):
        self.alpha = torch.tensor(alpha, requires_grad=True, device=device)
        self.rec_alpha = 0 < self.alpha < 1

        self.device = device 

        # Initialize instances of the models, could be done in other ways
        self.net0 = net().to(self.device)
        self.net1 = net().to(self.device)
        
        # self.test_steps = test_steps
        # os.makedirs(path_to_res, exist_ok=True)


        # self.path = path_to_res
        self.lamb = lamb


        self.dataset = dataset
        data, labels = [], []
        
        for batch in trainloader:
            img, l = batch['image'], batch['label']
            labels.append(l)
            data.append(img)
        self.data = torch.cat(data).to(self.device)
        self.labels = torch.cat(labels).to(self.device) # Added this myself

        # LOAD THE PARAMETERS INTO self.net0 and self.net1
        self.net0.load_state_dict(w0)
        self.net1.load_state_dict(wT)

        # DON'T TOUCH
        # We assume that labels have been restored separately, for details please refer to the paper.
        self.y = torch.cat(labels).to(device=self.device)
        # Dummy input.
        self.x = torch.normal(0, 1, size=self.data.shape, requires_grad=True, device=self.device)

        self.mean = torch.tensor(mean_std[0]).to(self.device).reshape(1, -1, 1, 1)
        self.std = torch.tensor(mean_std[1]).to(self.device).reshape(1, -1, 1, 1)
        # This is a trick (a sort of prior information) adopted from IG.
        prior_boundary(self.x, -self.mean / self.std, (1 - self.mean) / self.std)

    def reconstruction(self, eta, beta, iters, lr_decay, signed_grad=False, save_figure=True, save_interval=10, gif_path="reconstruction.gif"):
        # when taking the SME strategy, alpha is set within (0, 1).
        if 0 < self.alpha < 1:
            self.alpha.grad = torch.tensor(0.).to(self.device)
        optimizer = torch.optim.Adam(params=[self.x], lr=eta)
        alpha_opti = torch.optim.Adam(params=[self.alpha], lr=beta)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[iters // 2.667,
                                                                     iters // 1.6,
                                                                     iters // 1.142],
                                                         gamma=0.1)
        alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(alpha_opti,
                                                               milestones=[iters // 2.667,
                                                                           iters // 1.6,
                                                                           iters // 1.142],
                                                               gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        # Direction of the weight update.
        w1_w0 = []
        for p0, p1 in zip(self.net0.parameters(), self.net1.parameters()):
            w1_w0.append(p0.data - p1.data)
        norm = compute_norm(w1_w0)
        w1_w0 = [p / norm for p in w1_w0]

        # Construct the model for gradient inversion attack.
        require_grad(self.net0, False)
        require_grad(self.net1, False)
        with torch.no_grad():
            _net = copy.deepcopy(self.net0)
            for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                z.data = (1 - self.alpha) * p + self.alpha * q

        # Reconstruction
        _net.eval()
        stats = []
        images = []  # List to store intermediate images for the GIF
        with torch.no_grad():
            img_np = (self.x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np)) # Save the initial image
        for i in tqdm(range(iters)):
            optimizer.zero_grad()
            alpha_opti.zero_grad(set_to_none=False)
            _net.zero_grad()

            if self.rec_alpha:
                # Update the surrogate model.
                with torch.no_grad():
                    for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                        z.data = (1 - self.alpha) * p + self.alpha * q
            pred = _net(self.x)
            loss = criterion(input=pred, target=self.y)
            grad = torch.autograd.grad(loss, _net.parameters(), create_graph=True)
            norm = compute_norm(grad)
            grad = [p / norm for p in grad]

            # Compute x's grad.
            cos_loss = 1 - sum([
                p.mul(q).sum() for p, q in zip(w1_w0, grad)
            ])
            loss = cos_loss + self.lamb * total_variation(self.x)
            loss.backward()
            if signed_grad:
                self.x.grad.sign_()

            # Compute alpha's grad.
            if self.rec_alpha:
                with torch.no_grad():
                    for p, q, z in zip(self.net0.parameters(), self.net1.parameters(), _net.parameters()):
                        self.alpha.grad += z.grad.mul(
                            q.data - p.data
                        ).sum()
                if signed_grad:
                    self.alpha.grad.sign_()

            # Update x and alpha.
            optimizer.step()
            alpha_opti.step()
            prior_boundary(self.x, -self.mean / self.std, (1 - self.mean) / self.std)
            prior_boundary(self.alpha, 0, 1)
            if lr_decay:
                scheduler.step()
                alpha_scheduler.step()

            if save_figure:
                # Define dynamic save intervals
                if i < iters // 10:  # Save more frequently in the first half
                    current_save_interval = save_interval  # Use the default save_interval
                else:  # Save less frequently in the second half
                    current_save_interval = save_interval * 5  # Save every 30 iterations (or adjust as needed)

                # Save the image if the current iteration matches the interval
                if i % current_save_interval == 0:
                    with torch.no_grad():
                        img = self.x * self.std + self.mean  # Denormalize the image
                        img = img.clamp(0, 1)  # Clamp values to [0, 1]
                        img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Convert to numpy
                        images.append(Image.fromarray(img_np))  # Convert to PIL Image

        # Save the GIF

        if save_figure and images:
            # Duplicate the first frame to make it appear longer
            num_repeats_first_frame = 10  # Number of times to repeat the first frame
            num_repeats_per_frame = 3  # Number of times to repeat each of the first 10 frames

            extended_images = []

            # Extend the first image
            extended_images.extend([images[0]] * num_repeats_first_frame)  # Repeat the first image

            # Extend the first 10 images
            for img in images[1:10]:  # Loop through the first 10 images (excluding the first)
                extended_images.extend([img] * num_repeats_per_frame)  # Repeat each image

            # Add the rest of the images without duplication

            # Add the rest of the images without duplication
            extended_images.extend(images[10:])

            # Save the GIF with the extended frames
            extended_images[0].save(
                gif_path,
                save_all=True,
                append_images=extended_images[1:],  # Append the rest of the frames
                duration=100,  # Duration for each frame in milliseconds
                loop=0  # Infinite loop
            )
        return self.x, self.data, self.labels

            


        #     if i % self.test_steps == 0 or i == iters - 1:
        #         with torch.no_grad():
        #             _x = self.x * self.std + self.mean
        #             _data = self.data * self.std + self.mean
        #         measurement = psnr(_data, _x, sort=True)
        #         print(f"iter: {i}| alpha: {self.alpha.item():.2f}| (1 - cos): {cos_loss.item():.3f}| "
        #               f"psnr: {measurement:.3f}")
        #         stats.append({
        #             "iter": i,
        #             "alpha": self.alpha.item(),
        #             "cos_loss": cos_loss.item(),
        #             "psnr": measurement,
        #         })
        #         if save_figure:
        #             save_figs(tensors=_x, path=self.path, subdir=str(i), dataset=self.dataset)
        # if save_figure:
        #     save_figs(tensors=self.data * self.std + self.mean,
        #               path=self.path, subdir="original", dataset=self.dataset)
        # return stats
    

