import torch
from torch.nn import GaussianNLLLoss as TorchGaussianNLLLoss


class GaussianNLLLoss(TorchGaussianNLLLoss):
    def __init__(self, include_semiquant=True, fill_value=None):
        """
        Class for calculating Gaussian NLL loss, with various options.

        Parameters
        ----------
        include_semiquant : bool, default=True
            Whether to include semi-quantitative samples in the loss
        fill_value : float, optional
            If provided, use this value as the uncertainty for all
            semiquant predictions
        """
        # No reduction so we can apply masking if desired
        super().__init__(reduction="none")

        self.include_semiquant = include_semiquant
        self.fill_value = fill_value

    def forward(self, input, target, in_range, uncertainty):
        """
        Loss calculation

        Parameters
        ----------
        input : torch.Tensor
            Model prediction
        target : torch.Tensor
            Prediction target
        in_range : torch.Tensor
            `target`'s presence in the dynamic range of the assay. Give a value
            of < 0 for `target` below lower bound, > 0 for `target` above upper
            bound, and 0 or None for inside range
        uncertainty : torch.Tensor
            Uncertainty in `target` measurements

        Returns
        -------
        """
        # Clone to avoid modifying the original uncertainty measurements
        uncertainty_clone = uncertainty.clone()
        # Fill in semiquant values
        if self.include_semiquant and (self.fill_value is not None):
            idx = [r != 0 for r in in_range]
            uncertainty_clone[idx] = self.fill_value

        # Calculate loss (need to square uncertainty to convert to variance)
        loss = super().forward(input, target, uncertainty_clone**2)

        # Mask out losses for all semiquant measurements
        if not self.include_semiquant:
            mask = torch.tensor(
                [r == 0 for r in in_range], dtype=loss.dtype, device=loss.device
            )
            loss *= mask

        return loss.sum()
