from torch.nn.modules.loss import MSELoss as TorchMSELoss

class MSELoss(TorchMSELoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target, in_range):
        ## Calculate all MSE losses and then multiple each one by `not in_range`