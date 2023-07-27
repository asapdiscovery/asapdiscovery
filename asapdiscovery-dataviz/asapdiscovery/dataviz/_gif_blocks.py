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

    # 7ENE
    view_coords_MERS_CoV_Mpro_7ene = (
        0.710110664,
        0.317291290,
        -0.628544748,
        -0.485409290,
        -0.426031262,
        -0.763462067,
        -0.510019004,
        0.847245276,
        -0.148514912,
        0.000047840,
        0.000042381,
        -58.146995544,
        -1.610392451,
        -9.301478386,
        57.779785156,
        13.662354469,
        102.618484497,
        -20.000000000,
    )

    # 272
    view_coords_MERS_CoV_Mpro_272 = (
        0.781453907,
        0.495069802,
        -0.379777491,
        0.348743975,
        -0.851257741,
        -0.392085701,
        -0.517399728,
        0.173951998,
        -0.837877095,
        0.000047162,
        0.000063360,
        -106.999412537,
        6.953200340,
        11.497907639,
        42.910934448,
        -49050.539062500,
        49264.585937500,
        -20.000000000,
    )

    # mac1
    view_coords_SARS_CoV_2_Mac1 = (
        0.058132507,
        -0.604144037,
        -0.794748724,
        -0.915980995,
        0.284294397,
        -0.283112556,
        0.396988422,
        0.744434655,
        -0.536857843,
        0.000011945,
        -0.000066146,
        -87.475425720,
        13.680676460,
        25.153524399,
        -0.173997879,
        -1329.627441406,
        1504.635498047,
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
        "subP1": "140-145+163+172",
        "subP1_prime": "25-27",
        "subP2": "41+49+54",
        "subP3_4_5": "165-168+189-192",
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
        "nucleotide": "paleyellow",
        "bridge": "darksalmon",
        "phosphate": "brightorange",
        "anion_hole": "slate",
    }
