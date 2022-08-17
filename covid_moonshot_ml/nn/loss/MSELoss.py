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
        super(MSELoss, self).__init__()

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

    def forward(self, *args, **kwargs):
        """
        Dispatch method for calculating loss. All arguments are passed to the
        appropriate loss calculation function based on `self. loss_type`.
        """

        return self.loss_function(*args, **kwargs)

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
        in_range : int
            `target`'s presence in the dynamic range of the assay. Give a value
            of < 0 for `target` below lower bound, > 0 for `target` above upper
            bound, and 0 or None for inside range

        Returns
        -------
        torch.Tensor
            Calculated loss
        """
        ### Need to redo this because this onlny works if the input has a batch
        ###  size of 1
        ## Calculate loss
        loss = super(MSELoss, self).forward(input, target)

        ## If no value given for `in_range` or input is inside the assay range
        if in_range == 0:
            return loss

        ## If the target value is below the lower bound of the assay range,
        ##  only compute loss if input is inside range
        if in_range < 0:
            return (target > input) * loss

        ## If the target value is above the upper bound of the assay range,
        ##  only compute loss if input is inside range
        if in_range > 0:
            return (target < input) * loss

    def uncertainty_loss(self, input, target, uncertainty):
        """
        Uncertainty MSE loss calculation.

        Parameters
        ----------

        Returns
        -------
        """
