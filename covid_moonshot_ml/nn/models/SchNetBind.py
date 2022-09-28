import numpy as np
from torch_geometric.nn import SchNet


class SchNetBind(SchNet):
    """
    Light wrapper over the SchNet model from e3nn to compute a binding affinity
    of a ligand.
    """

    def __init__(self, *args, **kwargs):
        super(SchNetBind, self).__init__(*args, **kwargs)

    # TODO: Does not match signature in base class
    #  https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.SchNet.forward
    def forward(self, z, pos, lig):
        """
        Forward pass through the model. Each forward pass through this class
        makes two forward calls to the back-end model, predicting an energy for
        the bound complex and the combined energy of the protein and ligand
        separate.

        Parameters
        ----------
        z : torch.tensor
            Atomic number for each atom
        pos : torch.tensor
            Atomic position vectors
        lig : torch.tensor
            Boolean labels telling which atoms are part of the ligand

        Returns
        -------
        torch.tensor
            Binding affinity (pIC50) prediction
        """
        ## First make forward pass for the complex structure
        e_complex = super(SchNetBind, self).forward(z, pos)

        ## Make a copy of the position vector and move the ligand molecules 100A
        ##  away from its original position
        # pos shouldn't require grad but just make sure
        new_pos = pos.detach().clone()
        new_pos[lig, :] += 100

        ## Calculate total energy of the ligand and Mpro separately
        e_sep = super(SchNetBind, self).forward(z, new_pos)

        ## Need to adjust the units (pred is in eV, labels are in -log10(K_D))
        ## dG = kTln(K_D)
        ## dG/kT = log10(K_D)/log10(e)
        ## -log10(K_D) = -log10(e)/kT * dG
        ## [dG] = eV (from SchNet)
        ## kt = 25.7 meV = 25.7e-3 eV
        dG = e_complex - e_sep
        target_pred = -dG / (25.7e-3) * (np.log10(np.e))

        return target_pred
