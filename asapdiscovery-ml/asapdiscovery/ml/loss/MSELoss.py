import torch
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
        mask = torch.tensor(
            [
                1.0 if r == 0 else ((r < 0) == (t < i))
                for i, t, r in zip(input, target, in_range)
            ]
        )
        mask = mask.to(loss.device)

        return (mask * loss).mean()
