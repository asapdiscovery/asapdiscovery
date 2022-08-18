import torch
from torch.nn.modules.loss import MSELoss as TorchMSELoss


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
        ## No reduction so we can apply whatever adjustment to each sample
        super(MSELoss, self).__init__(reduction="none")

        if loss_type is not None:
            loss_type = loss_type.lower()
            if loss_type == "step":
                self.loss_function = self.step_loss
            elif loss_type == "uncertainty":
                self.loss_function = self.uncertainty_loss
            else:
                raise ValueError(f'Unknown loss_type "{loss_type}"')
        else:
            self.loss_function = super(MSELoss, self).forward

        self.loss_type = loss_type

    def forward(self, input, target, in_range, uncertainty):
        """
        Dispatch method for calculating loss. All arguments should be passed
        regardless of actual loss function to keep an identical signature for
        this class. Data is passed to `self.loss_function`.
        """

        if self.loss_type is None:
            return self.loss_function(input, target).mean()
        elif self.loss_type == "step":
            return self.loss_function(input, target, in_range)
        elif self.loss_type == "uncertainty":
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
        ## Calculate loss
        loss = super(MSELoss, self).forward(input, target)

        ## Calculate mask
        mask = []
        for i in range(len(in_range)):
            ## If input is inside the assay range
            if in_range[i] == 0:
                m = 1.0
            ## If the target value is below the lower bound of the assay range,
            ##  only compute loss if input is inside range
            elif in_range[i] < 0:
                m = target[i] < input[i]
            ## If the target value is above the upper bound of the assay range,
            ##  only compute loss if input is inside range
            elif in_range[i] > 0:
                m = target[i] > input[i]
            mask.append(m)
        mask = torch.tensor(mask)

        return (mask * loss).mean()

    def uncertainty_loss(self, input, target, uncertainty):
        """
        Uncertainty MSE loss calculation. Loss for each sample is calculated by
        first dividing the difference between `input` and `target` by the
        uncertainty in the `target` measurement.

        Parameters
        ----------
        input : torch.Tensor
            Model prediction
        target : torch.Tensor
            Prediction target
        uncertainty : torch.Tensor
            Uncertainty in `target` measurements

        Returns
        -------
        """
        ## Calculate loss
        loss = super(MSELoss, self).forward(input, target)

        ## Divide by uncertainty squared
        loss /= uncertainty**2

        return loss.mean()
