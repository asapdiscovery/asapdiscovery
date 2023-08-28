import torch
from asapdiscovery.ml.loss import MSELoss


class GroupedMSELoss(MSELoss):
    def __init__(self, loss_type=None):
        """
        Class for calculating MSE loss for GroupedModel predictions, and appropriately
        setting the gradients. Use MSELoss class to actually calculate the loss
        (with options), while this class performs the appropriate gradient updates.

        Parameters
        ----------
        loss_type : str, optional
            Which type of loss to use:
             * None: vanilla MSE
             * "step": step MSE loss
             * "uncertainty": MSE loss with added uncertainty
        """
        # No reduction so we can apply whatever adjustment to each sample
        super().__init__(loss_type=loss_type)

    def forward(self, model, input, target, in_range, uncertainty):
        """
        Dispatch method for calculating loss. All arguments should be passed
        regardless of actual loss function to keep an identical signature for
        this class. Data is passed to `self.loss_function`.
        """

        if self.loss_type is None:
            # Just need to calculate mean to get MSE
            if model.training:
                for p in model.parameters():
                    p.grad *= 2 * (input - target).mean(axis=None)
            return self.loss_function(input, target).mean()
        elif self.loss_type == "step":
            # Call step_loss
            if model.training:
                if (
                    (in_range == 0)
                    or ((in_range < 0) and (target < input))
                    or ((in_range > 1) and (input < target))
                ):
                    for p in model.parameters():
                        p.grad *= 2 * (input - target).mean(axis=None)
                else:
                    # Zero grads if we're not calculating loss for this example
                    model.zero_grad()
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
