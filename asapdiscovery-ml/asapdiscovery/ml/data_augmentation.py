from copy import deepcopy

import torch


class JitterFixed:
    """
    Jitter an input pose by drawing noise from a fixed normal distribution. Noise is
    generated per-atom.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        rand_seed: int | None = None,
        dict_key: str = "pos",
    ):
        """
        Parameters
        ----------
        mean : float
            Mean of noise distribution
        std : float
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


class JitterVariable:
    """
    Jitter an input pose by drawing noise from a normal distribution that is based on a
    property in the passed pose. Noise is generated per-atom.
    """

    def __init__(
        self,
        mean: callable,
        std: callable,
        rand_seed: int | None = None,
        pos_dict_key: str = "pos",
        temp_dict_key: str = "b",
    ):
        """
        Parameters
        ----------
        mean : callable
            Mean of noise distribution. The function will be used to transform the value
            stored in the temperature information into a mean for a normal distribution
        std : callable
            Standard deviation of noise distribution. The function will be used to
            transform the value stored in the temperature information into a standard
            deviation for a normal distribution
        rand_seed : int, optional
            Random seed for noise generation
        pos_dict_key : str, default="pos"
            Key to access the coords in input dict
        temp_dict_key : str, default="b"
            Key to access temperature/B factors in input dict
        """

        self.mean = mean
        self.std = std
        self.rand_seed = rand_seed
        if rand_seed is not None:
            self.g = torch.Generator().manual_seed(rand_seed)
        else:
            self.g = torch.Generator().manual_seed(torch.random.seed())
        self.pos_dict_key = pos_dict_key
        self.temp_dict_key = temp_dict_key

    def __call__(self, pose, inplace=False):
        """
        Apply noise to each atom. Unless inplace is True, this method will create a copy
        of the input coordinate Tensor.

        Parameters
        ----------
        pose : dict
            Pose dict containing (at minimum) the initial coordinates to be jittered and
            the temperature information. Noise will be generated independently for each
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
            self.mean(pose_copy[self.temp_dict_key])
            .reshape((-1, 1))
            .broadcast_to(pose_copy[self.pos_dict_key].shape)
            .to("cpu")
        )
        std = (
            self.std(pose_copy[self.temp_dict_key])
            .reshape((-1, 1))
            .broadcast_to(pose_copy[self.pos_dict_key].shape)
            .to("cpu")
        )

        # Generate noise (the std will be broadcast to the same shape as mean)
        noise = torch.normal(mean=mean, std=std, generator=self.g)
        # Add the noise
        pose_copy[self.pos_dict_key] += noise.to(pose_copy[self.pos_dict_key].device)

        return pose_copy
