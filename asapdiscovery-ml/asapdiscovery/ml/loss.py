import torch
from torch.nn import GaussianNLLLoss as TorchGaussianNLLLoss
from torch.nn import MSELoss as TorchMSELoss


class MSELoss(TorchMSELoss):
    def __init__(self, loss_type=None):
        """
        Class for calculating MSE loss, with various options

        Parameters
        ----------
        loss_type : str, optional
            Which type of loss to use:
             * None: vanilla MSE
             * "step": step MSE loss
             * "uncertainty": MSE loss with added uncertainty
        """
        # No reduction so we can apply whatever adjustment to each sample
        super().__init__(reduction="none")

        if loss_type is not None:
            loss_type = loss_type.lower()
            if loss_type == "step":
                self.loss_function = self.step_loss
            elif loss_type == "uncertainty":
                self.loss_function = self.uncertainty_loss
            else:
                raise ValueError(f'Unknown loss_type "{loss_type}"')
        else:
            self.loss_function = super().forward

        self.loss_type = loss_type

    def forward(self, input, target, in_range, uncertainty):
        """
        Dispatch method for calculating loss. All arguments should be passed
        regardless of actual loss function to keep an identical signature for
        this class. Data is passed to `self.loss_function`.
        """

        if self.loss_type is None:
            # Just need to calculate mean to get MSE
            return self.loss_function(input, target).mean()
        elif self.loss_type == "step":
            # Call step_loss
            return self.loss_function(input, target, in_range)
        elif self.loss_type == "uncertainty":
            # Call uncertainty_loss
            return self.loss_function(input, target, uncertainty)

    def step_loss(self, input, target, in_range):
        """
        Step loss calculation. For `in_range` < 0, loss is returned as 0 if
        `input` < `target`, otherwise MSE is calculated as normal. For
        `in_range` > 0, loss is returned as 0 if `input` > `target`, otherwise
        MSE is calculated as normal. For `in_range` == 0, MSE is calculated as
        normal.

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

        Returns
        -------
        torch.Tensor
            Calculated loss
        """
        # Calculate loss
        loss = super().forward(input, target)

        # Calculate mask:
        #  1.0 - If input or data is semiquant and prediction is inside the
        #    assay range
        #  0.0 - If data is semiquant and prediction is outside the assay range
        # r < 0 -> measurement is below thresh, want to count if pred > target
        # r > 0 -> measurement is above thresh, want to count if pred < target
        mask = torch.tensor(
            [
                1.0 if r == 0 else ((r < 0) == (t < i))
                for i, t, r in zip(input, target, in_range)
            ]
        )
        mask = mask.to(input.device)

        # Need to add the max in the denominator in case there are no values that we
        #  want to calculate loss for
        loss = (loss * mask) / max(torch.sum(mask), 1)

        return loss


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


class RangeLoss(torch.nn.Module):
    def __init__(self, lower_lim, upper_lim):
        """
        Class for calculating a loss to penalize predictions outside of the given range.
        Current implementation uses a squared difference penalty.

        Parameters
        ----------
        lower_lim : float
            Bottom limit of acceptable range
        upper_lim : float
            Upper limit of acceptable range
        """
        super().__init__()

        self.lower_lim = lower_lim
        self.upper_lim = upper_lim

    def forward(self, input, target, in_range, uncertainty):
        """
        No loss for predictions within self range, otherwise calculate squared distance
        to closest bound.

        Parameters
        ----------
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
        torch.Tensor
            Calculated loss
        """
        if input < self.lower_lim:
            return torch.pow(input - self.lower_lim, 2)
        elif input > self.upper_lim:
            return torch.pow(input - self.upper_lim, 2)
        else:
            return input * 0
