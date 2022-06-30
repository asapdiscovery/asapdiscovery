from torch.nn.modules.loss import MSELoss as TorchMSELoss

class MSELoss(TorchMSELoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target, in_range=None):
        """
        Loss calculation.

        Parameters
        ----------
        input : torch.Tensor
            Model prediction
        target : torch.Tensor
            Prediction target
        in_range : float, optional
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

        ## If no value given for `in_range` or input is inside the assay range
        if (in_range is None) or (in_range == 0):
            return(loss)

        ## If the target value is below the lower bound of the assay range,
        ##  only compute loss if input is inside range
        if in_range < 0:
            return((target > input) * loss)

        ## If the target value is above the upper bound of the assay range,
        ##  only compute loss if input is inside range
        if in_range > 0:
            return((target < input) * loss)
