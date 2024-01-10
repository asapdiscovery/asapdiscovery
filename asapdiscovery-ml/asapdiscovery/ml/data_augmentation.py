import torch


class Jitter:
    """
    Jitter an input pose by drawing noise from a normal distribution. Noise is generated
    per-atom.
    """

    def __init__(self, mean: float, std: float, rand_seed: int = None):
        """
        Parameters
        ----------
        mean : float
            Mean of noise distribution
        std : float
            Standard deviation of noise distribution
        rand_seed : int, optional
            Random seed for noise generation
        """

        self.mean = mean
        self.std = std
        self.rand_seed = rand_seed
        if rand_seed is not None:
            self.g = torch.Generator().manual_seed(rand_seed)
        else:
            self.g = torch.Generator().manual_seed(torch.random.seed())

    def __call__(self, coords, inplace=False):
        """
        Apply noise to each atom. Unless inplace is True, this method will create a copy
        of the input coordinate Tensor.

        Parameters
        ----------
        coords : torch.Tensor
            Initial coordinates to be jittered. Noise will be generated independently
            for each
        inplace : bool, default=False
            Modify the passed Tensor in place, rather than first copying

        Returns
        -------
        torch.Tensor
            Jittered coordinates
        """

        # Fist make a copy of the input coords (if inplace is False)
        if not inplace:
            coords_copy = coords.clone().detach()
        else:
            # Should just be a reference so inputs should get modified
            coords_copy = coords
        # Create the mean Tensor, which should just have the same shape as the coords
        mean = torch.full_like(coords_copy, self.mean).to("cpu")
        # Generate noise (the std will be broadcast to the same shape as mean)
        noise = torch.normal(mean=mean, std=self.std, generator=self.g)
        # Add the noise
        coords_copy += noise.to(coords_copy.device)

        return coords_copy
