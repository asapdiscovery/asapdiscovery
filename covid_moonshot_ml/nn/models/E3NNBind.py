from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network
import numpy as np

class E3NNBind(Network):
    """docstring for E3NNBind"""
    def __init__(self, *args, **kwargs):
        super(E3NNBind, self).__init__(*args, **kwargs)

    def forward(self, d):
        ## First make forward pass for the complex structure
        e_complex = super(E3NNBind, self).forward(d)

        ## Make a copy of the position vector and move the ligand molecules 100A
        ##  away from its original position
        # pos shouldn't require grad but just make sure
        new_pos = d['pos'].detach().clone()
        new_pos[d['lig'],:] += 100
        d_new = {'pos': new_pos, 'x': d['x'], 'z': d['z']}

        ## Calculate total energy of the ligand and Mpro separately
        e_sep = super(E3NNBind, self).forward(d_new)

        ## Need to adjust the units (pred is in eV, labels are in -log10(K_D))
        ## dG = kTln(K_D)
        ## dG/kT = log10(K_D)/log10(e)
        ## -log10(K_D) = -log10(e)/kT * dG
        ## [dG] = eV (from SchNet)
        ## kt = 25.7 meV = 25.7e-3 eV
        dG = e_complex - e_sep
        target_pred = -dG/(25.7e-3)*(np.log10(np.e))

        return(target_pred)
