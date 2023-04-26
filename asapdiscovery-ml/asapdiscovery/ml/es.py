"""
Class for handling early stopping in training.
"""
from copy import deepcopy


class BestEarlyStopping:
    """
    Class for handling early stopping in training based on improvement over best loss.
    """

    def __init__(self, patience):
        """
        Parameters
        ----------
        patience : int, optional
            The maximum number of epochs to continue training with no improvement in the
            val loss. If not given, no early stopping will be performed
        """
        super().__init__()
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
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
            return False

        # Update best loss and best weights
        if loss < self.best_loss:
            self.best_loss = loss
            # Need to deepcopy so it doesn't update with the model weights
            self.best_wts = deepcopy(wts_dict)
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


class ConvergedEarlyStopping:
    """
    Class for handling early stopping in training based on whether loss is still
    changing. Check that all differences of the past n losses from the average of
    those losses are within tolerance.
    """

    def __init__(self, n_check, divergence):
        """
        Parameters
        ----------
        n_check : int
            Number of past epochs to keep track of when calculating divergence
        divergence : float
            Max allowable difference from the mean of the losses as a fraction of the
            average loss
        """
        super().__init__()
        self.n_check = n_check
        self.divergence = divergence

        # Variables to track early stopping
        self.losses = []

    def check(self, loss):
        """
        Check if training should be stopped. Return True to stop, False to keep going.

        Parameters
        ----------
        loss : float
            Loss from the previous training epoch

        Returns
        -------
        bool
            Whether to stop training
        """
        import numpy as np

        # Add most recent loss
        self.losses += [loss]

        # Don't have enough samples yet, so keep training
        if len(self.losses) < self.n_check:
            return False

        # Full loss buffer, so get rid of earliest loss
        if len(self.losses) > self.n_check:
            self.losses = self.losses[1:]

        # Check for early stopping
        mean_loss = np.mean(self.losses)
        all_abs_diff = np.abs(np.asarray(self.losses) - mean_loss)

        return np.all(all_abs_diff < (self.divergence * mean_loss))
