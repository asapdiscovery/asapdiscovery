"""
Class for handling early stopping in training.
"""


class EarlyStopping(object):
    """Class for handling early stopping in training."""

    def __init__(self, patience):
        """
        Parameters
        ----------
        patience : int, optional
            The maximum number of epochs to continue training with no improvement in the
            val loss. If not given, no early stopping will be performed
        """
        super(EarlyStopping, self).__init__()
        self.patience = patience

        # Variables to track early stopping
        self.counter = 0
        self.best_loss = None
        self.best_wts = None
        self.best_epoch = 0

    def check(self, epoch, loss, wts_dict):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Model loss from the current epoch of training
        wts_dict : dict
            Weights dict from Pytorch for keeping track of the best model

        Returns
        -------
        bool
            Whether to stop training
        """
        # If this is the first epoch, just set internal variables and return
        if self.best_loss is None:
            self.best_loss = loss
            self.best_wts = wts_dict
            return False

        # Update best loss and best weights
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_wts = wts_dict
            self.best_epoch = epoch

            # Reset counter
            self.counter = 0

            # Keep training
            return False

        # Increment counter and check for stopping
        self.counter += 1
        if self.counter == self.patience:
            return True

        return False
