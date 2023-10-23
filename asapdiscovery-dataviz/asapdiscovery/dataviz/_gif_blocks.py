from .viz_targets import VizTargets

"""
Find these by going through GIF generation with `pse_share=True`, then inspecting
session_5.pse with pymol, orienting and running `get_view` in pymol terminal.
"""


class GIFBlockData:
    @classmethod
    def get_view_coords(cls):
        """
        Get the master orient coords.
        """
        return getattr(cls, "master_view_coords")

    @classmethod
    def get_pocket_dict(cls, target):
        target_name = VizTargets.get_target_name(target, underscore=True)
        return getattr(cls, f"pocket_dict_{target_name}")

    @classmethod
    def get_color_dict(cls, target):
        protein_name = VizTargets.get_protein_name(target, underscore=True)
        return getattr(cls, f"color_dict_{protein_name}")

    master_view_coords = (\
    -0.735560179,   -0.330644876,   -0.591291904,\
     0.100930467,   -0.916550815,    0.386970639,\
    -0.669899166,    0.224961400,    0.707550228,\
    -0.000009850,    0.000009298,  -69.100456238,\
    12.334323883,   -0.101311684,   20.903310776,\
   -66.815620422,  205.012481689,  -20.000000000 )

    # set colorings of subpockets by resn. This may change over time.,
    # first define the residues that span the subpockets
    # SARS2
    pocket_dict_SARS_CoV_2_Mpro = {
        "subP1": "140+141+142+143+144+145+163+172",
        "subP1_prime": "25+26+27",
        "subP2": "41+49+54",
        "subP3_4_5": "165+166+167+168+189+190+191+192",
        "sars_unique": "25+49+142+164+168+169+181+186+188+190+191",
    }

    # MERS
    pocket_dict_MERS_CoV_Mpro = {
        "subP1": "143+144+145+146+147+148+166+175",
        "subP1_prime": "25+26+27",
        "subP2": "41+49+54",
        "subP3_4_5": "168+169+170+171+192+193+194+195",
        "sars_unique": "25+49+145+167+171+172+184+189+191+193+194",
    }

    # MAC1
    pocket_dict_SARS_CoV_2_Mac1 = {
        "nucleotide": "154+156+22+23+24+52+49+125",
        "bridge": "126+155",
        "phosphate": "46+47+48+38+39+40+130+131+132+127+128+97",
        "anion_hole": "129+157+160+136+164",
    }

    # now define the colors per subpocket
    color_dict_Mpro = {
        "subP1": "yellow",
        "subP1_prime": "orange",
        "subP2": "skyblue",
        "subP3_4_5": "aquamarine",
    }

    color_dict_Mac1 = {
        "nucleotide": "yellow",
        "bridge": "pink",
        "phosphate": "orange",
        "anion_hole": "blue",
    }
