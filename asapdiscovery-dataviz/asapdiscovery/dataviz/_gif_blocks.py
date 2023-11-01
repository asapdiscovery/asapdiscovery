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

    master_view_coords = (
        -0.735560179,
        -0.330644876,
        -0.591291904,
        0.100930467,
        -0.916550815,
        0.386970639,
        -0.669899166,
        0.224961400,
        0.707550228,
        -0.000009850,
        0.000009298,
        -69.100456238,
        12.334323883,
        -0.101311684,
        20.903310776,
        -66.815620422,
        205.012481689,
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

    # 3CPRO
    pocket_dict_EV_D68_3Cpro = {
        "subP1": "23+24+25+145+112+106+107+108",
        "subP1_prime": "161+166+141+142+143+144+147",
        "subP2": "130+131+132+133+69+71+39+40",
        "subP3": "127+128+129+162+163+164+165",
        "subP4" : "122+124+125+126+170",
    }

    pocket_dict_EV_A71_3Cpro = {
        "subP1": "126+125+122+170+163+164+165",
        "subP1_prime": "162+127+128+130+129+131",
        "subP2": "20+23+25+24+106+147+145",
        "subP3": "21+22+40+38+71+69",
        "subP4" : "42+43+39+73+59+60+61",
    }

    pocket_dict_ZIKV_NS2B_NS3pro = {
        "subP1": "151+161+129+160+150+130+131+132+135+134",
        "subP1_prime": "54+51+52+36+35+133",
        "subP2": "75+72+81+83+152",
        "subP3": "153+154+155+86+85+84",
    }

    pocket_dict_DENV_NS2B_NS3pro = {
        "subP1": "135+151+161+159+129+150+130+132+131",
        "subP1_prime": "51+52+36+38+54",
        "subP2": "83+82+81+72+50+75+152",
        "subP3": "85+84+153+154+155+86",
    }

    # now define the colors per subpocket for each target (cross-variant)
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
    color_dict_3Cpro = {
        "subP1": "orange",
        "subP1_prime": "yellow",
        "subP2": "blue",
        "subP3": "cyan",
        "subP4" : "magenta"
    }
    color_dict_NS3pro = {
        "subP1": "yellow",
        "subP1_prime": "orange",
        "subP2": "blue",
        "subP3": "cyan",
    }