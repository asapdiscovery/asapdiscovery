from copy import deepcopy

import torch


class JitterFixed:
    """
    Jitter an input pose by drawing noise from a fixed normal distribution. Noise is
    generated per-atom.
    """

    def __init__(
        self,
        mean: float = 1,
        std: float = 0.1,
        rand_seed: int | None = None,
        dict_key: str = "pos",
    ):
        """
        Parameters
        ----------
        mean : float, default=1
            Mean of noise distribution
        std : float, default=0.1
            Standard deviation of noise distribution
        rand_seed : int, optional
            Random seed for noise generation
        dict_key : str, default="pos"
            If the inputs are a dict, this will be the key used to access the coords in
            the dict
        """

        self.mean = mean
        self.std = std
        self.rand_seed = rand_seed
        if rand_seed is not None:
            self.g = torch.Generator().manual_seed(rand_seed)
        else:
            self.g = torch.Generator().manual_seed(torch.random.seed())
        self.dict_key = dict_key

    def __call__(self, coords, inplace=False):
        """
        Apply noise to each atom. Unless inplace is True, this method will create a copy
        of the input coordinate Tensor.

        Parameters
        ----------
        coords : torch.Tensor | dict
            Initial coordinates to be jittered. Noise will be generated independently
            for each
        inplace : bool, default=False
            Modify the passed Tensor in place, rather than first copying

        Returns
        -------
        torch.Tensor
            Jittered coordinates
        """

        # Figure out if we're working with a dict or raw Tensor inputs
        if isinstance(coords, dict):
            dict_inp = True
        else:
            dict_inp = False

        # Fist make a copy of the input coords (if inplace is False)
        if not inplace:
            if dict_inp:
                dict_copy = deepcopy(coords)
                coords_copy = dict_copy[self.dict_key]
            else:
                coords_copy = coords.clone().detach()
        else:
            # Should just be a reference so inputs should get modified
            if dict_inp:
                dict_copy = coords
                coords_copy = coords_copy[self.dict_key]
            else:
                coords_copy = coords
        # Create the mean Tensor, which should just have the same shape as the coords
        mean = torch.full_like(coords_copy, self.mean).to("cpu")
        # Generate noise (the std will be broadcast to the same shape as mean)
        noise = torch.normal(mean=mean, std=self.std, generator=self.g)
        # Add the noise
        coords_copy += noise.to(coords_copy.device)

        if dict_inp:
            return dict_copy
        else:
            return coords_copy


class JitterBFactor:
    """
    Jitter an input pose by drawing noise from a normal distribution that is based on
    its B factor. Noise is generated per-atom.
    """

    def __init__(
        self,
        rand_seed: int | None = None,
        pos_dict_key: str = "pos",
        b_dict_key: str = "b",
    ):
        """
        Parameters
        ----------
        rand_seed : int, optional
            Random seed for noise generation
        pos_dict_key : str, default="pos"
            Key to access the coords in input dict
        b_dict_key : str, default="b"
            Key to access B factors in input dict
        """

        self.rand_seed = rand_seed
        if rand_seed is not None:
            self.g = torch.Generator().manual_seed(rand_seed)
        else:
            self.g = torch.Generator().manual_seed(torch.random.seed())
        self.pos_dict_key = pos_dict_key
        self.b_dict_key = b_dict_key

    @staticmethod
    def b_factor_to_mean(b):
        """
        Convert B factors to a mean that can be used to draw noise from.
        The math in this function is based on the defintion of the B factor as
        B = (8(pi^2)/3) <u^2>
        We will approximate the mean of our Gaussian as the RMSD = sqrt(<u^2>):
        sqrt(<u^2>) = sqrt(3B / (8(pi^2)))

        Parameters
        ----------
        b : torch.Tensor
            Tensor of B factors

        Returns
        -------
        torch.Tensor
            Tensor of converted RMSDs
        """
        return (3 * b / (8 * torch.pi**2)).sqrt()

    @staticmethod
    def b_factor_to_std(b):
        """
        Convert B factors to a standard ceviation that can be used to draw noise from.
        The math in this function is somewhat arbitrary, until we get a better idea. The
        20 scaling factor is based on the value that OpeneEye assigns to docked ligands
        (which is 20), and the square root is to convert from A^2 to A.

        Parameters
        ----------
        b : torch.Tensor
            Tensor of B factors

        Returns
        -------
        torch.Tensor
            Tensor of converted RMSDs
        """
        return (b / 20).sqrt()

    def __call__(self, pose, inplace=False):
        """
        Apply noise to each atom. Unless inplace is True, this method will create a copy
        of the input coordinate Tensor.

        Parameters
        ----------
        pose : dict
            Pose dict containing (at minimum) the initial coordinates to be jittered and
            the information. Noise will be generated independently for each
            atom
        inplace : bool, default=False
            Modify the passed dict in place, rather than first copying

        Returns
        -------
        dict
            Copy of the input pose with jittered coordinates
        """

        # Fist make a copy of the input pose (if inplace is False)
        if not inplace:
            pose_copy = deepcopy(pose)
        else:
            # Should just be a reference so inputs should get modified
            pose_copy = pose

        # Calculate mean and std values. Need to do some reshaping/broadcasting to make
        #  the shapes line up
        mean = (
            JitterBFactor.b_factor_to_mean(pose_copy[self.b_dict_key])
            .reshape((-1, 1))
            .broadcast_to(pose_copy[self.pos_dict_key].shape)
            .to("cpu")
        )
        std = (
            JitterBFactor.b_factor_to_std(pose_copy[self.b_dict_key])
            .reshape((-1, 1))
            .broadcast_to(pose_copy[self.pos_dict_key].shape)
            .to("cpu")
        )

        # Generate noise (the std will be broadcast to the same shape as mean)
        noise = torch.normal(mean=mean, std=std, generator=self.g)
        # Add the noise
        pose_copy[self.pos_dict_key] += noise.to(pose_copy[self.pos_dict_key].device)

        return pose_copy
