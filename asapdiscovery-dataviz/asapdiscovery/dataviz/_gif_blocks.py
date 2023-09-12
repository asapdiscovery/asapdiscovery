from .viz_targets import VizTargets

"""
Find these by going through GIF generation with `pse_share=True`, then inspecting
session_5.pse with pymol, orienting and running `get_view` in pymol terminal.
"""


class GIFBlockData:
    @classmethod
    def get_view_coords(cls, target):
        target_ = VizTargets.get_name_underscore(target)
        return getattr(cls, f"view_coords_{target_}")

    @classmethod
    def get_pocket_dict(cls, target):
        target_name = VizTargets.get_target_name(target, underscore=True)
        return getattr(cls, f"pocket_dict_{target_name}")

    @classmethod
    def get_color_dict(cls, target):
        protein_name = VizTargets.get_protein_name(target, underscore=True)
        return getattr(cls, f"color_dict_{protein_name}")

    view_coords_SARS_CoV_2_Mpro = (
        -0.339127213,
        -0.370405823,
        -0.864748597,
        0.307130545,
        -0.912446380,
        0.270390421,
        -0.889192164,
        -0.173893914,
        0.423199117,
        0.000072375,
        0.000199302,
        -80.242187500,
        6.715160370,
        92.460678101,
        114.338867188,
        -458.494476318,
        618.980895996,
        -20.000000000,
    )

    # MERS
    view_coords_MERS_CoV_Mpro = (
        -0.635950804,
        -0.283323288,
        -0.717838645,
        -0.040723491,
        -0.916550398,
        0.397835642,
        -0.770651817,
        0.282238036,
        0.571343124,
        0.000061535,
        0.000038342,
        -58.079559326,
        8.052228928,
        0.619271040,
        21.864795685,
        -283.040344238,
        399.190032959,
        -20.000000000,
    )

    # mac1
    view_coords_SARS_CoV_2_Mac1 = (
        0.199152365,
        -0.560011566,
        -0.804189026,
        -0.606427014,
        0.574200273,
        -0.550032735,
        0.769795358,
        0.597223163,
        -0.225250959,
        0.000011945,
        -0.000066146,
        -74.581398010,
        13.680676460,
        25.153524399,
        -0.173997879,
        -2000.212524414,
        2149.432861328,
        -20.000000000,
    )

    # mac1 monomers from 07-19 release
    view_coords_SARS_CoV_2_Mac1_monomer = (
        0.088525243,
        0.968614519,
        0.232240111,
        -0.460261583,
        0.246554896,
        -0.852860034,
        -0.883353055,
        -0.031390585,
        0.467653960,
        0.000071935,
        -0.000032585,
        -91.646080017,
        2.006669521,
        -8.887639046,
        14.542367935,
        64.035736084,
        119.277297974,
        -20.000000000,
    )

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
