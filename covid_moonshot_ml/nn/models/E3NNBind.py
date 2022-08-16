from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
import numpy as np


class E3NNBind(Network):
    """
    Light wrapper over the Network model from e3nn to compute a binding affinity
    of a ligand.
    """

    def __init__(self, *args, **kwargs):
        super(E3NNBind, self).__init__(*args, **kwargs)

    def forward(self, d):
        """
        Forward pass through the model. Each forward pass through this class
        makes two forward calls to the back-end model, predicting an energy for
        the bound complex and the combined energy of the protein and ligand
        separate.

        Parameters
        ----------
        d : dict[str->torch.tensor]
            Entry in data.dataset.DockedDataset to calculate binding affinity
            for. Should have the following entries:
            * 'pos': atom positions
            * 'x': atom features
            * 'lig': boolean ligand labels
            * 'z': node attributes, optional

        Returns
        -------
        torch.tensor
            Binding affinity (pIC50) prediction
        """
        ## First make forward pass for the complex structure
        e_complex = super(E3NNBind, self).forward(d)

        ## Make a copy of the position vector and move the ligand molecules 100A
        ##  away from its original position
        # pos shouldn't require grad but just make sure
        new_pos = d["pos"].detach().clone()
        new_pos[d["lig"], :] += 100
        d_new = {"pos": new_pos, "x": d["x"], "z": d["z"]}

        ## Calculate total energy of the ligand and Mpro separately
        e_sep = super(E3NNBind, self).forward(d_new)

        ## Need to adjust the units (pred is in eV, labels are in -log10(K_D))
        ## dG = kTln(K_D)
        ## dG/kT = log10(K_D)/log10(e)
        ## -log10(K_D) = -log10(e)/kT * dG
        ## [dG] = eV (from SchNet)
        ## kt = 25.7 meV = 25.7e-3 eV
        dG = e_complex - e_sep
        target_pred = -dG / (25.7e-3) * (np.log10(np.e))

        return target_pred
