from typing import Union

from asapdiscovery.data.services.postera.manifold_data_validation import (
    TargetProteinMap,
    TargetTags,
)


class GIFBlockData:
    @classmethod
    def get_view_coords(cls) -> tuple[float]:
        """
        Get the master orient coords.
        """
        return getattr(cls, "master_view_coords")

    @classmethod
    def get_pocket_dict(cls, target: Union[TargetTags, str]) -> dict[str, str]:
        if isinstance(target, TargetTags):
            target = str(target.value)
        else:
            target = str(target)
        # need underscored protein name for pocket_dict
        target = target.replace("-", "_")
        return getattr(cls, f"pocket_dict_{target}")

    @classmethod
    def get_color_dict(cls, target: TargetTags) -> dict[str, str]:
        if isinstance(target, TargetTags):
            target = str(target.value)
        else:
            target = str(target)
        protein_name = TargetProteinMap[target]
        # need underscored protein name for color_dict
        protein_name = protein_name.replace("-", "_")
        return getattr(cls, f"color_dict_{protein_name}")

    master_view_coords = (
        -0.711445928,
        -0.178503618,
        -0.679694653,
        -0.008906467,
        -0.964834511,
        0.262710690,
        -0.702686131,
        0.192958608,
        0.684835911,
        0.000000000,
        0.000000000,
        -93.089912415,
        13.605349541,
        -1.358839035,
        15.771842957,
        63.240909576,
        122.938896179,
        -20.000000000,
    )

    # set colorings of subpockets by resn. This may change over time.,
    # first define which chain the binding pocket lives in per target.
    pocket_dict_chains_per_target = {
        "SARS-CoV-2-Mpro": "A",
        "SARS-CoV-2-Mac1": "A",
        "SARS-CoV-2-N-protein": "A",
        "SARS-CoV-2-Mac1-monomer": "A",
        "MERS-CoV-Mpro": "A",
        "EV-D68-3Cpro": "A",
        "EV-A71-3Cpro": "A",
        "EV-A71-2Apro": "A",
        "EV-A71-Capsid": "A",
        "EV-D68-Capsid": "A",
        "DENV-NS2B-NS3pro": "B",
        "ZIKV-NS2B-NS3pro": "B",
        "ZIKV-RdRppro": "A",
    }

    # define the residues that span the subpockets
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

    # N-protein
    pocket_dict_SARS_CoV_2_N_protein = {
        "undecided": "1-116",
    }

    # 3CPRO
    pocket_dict_EV_D68_3Cpro = {
        "subP1": "23+24+25+145+112+106+107+108+22",
        "subP1_prime": "161+166+141+142+143+144+147+109",
        "subP2": "130+131+132+133+69+71+39+40",
        "subP3": "127+128+129+162+163+164+165",
        "subP4": "122+124+125+126+170",
    }

    pocket_dict_EV_A71_3Cpro = {
        "subP1": "125+122+170+163+164+165+142+143+144+161",
        "subP1_prime": "162+127+128+130+129+131+126",
        "subP2": "20+23+25+24+106+147+145",
        "subP3": "21+22+40+38+71+69",
        "subP4": "42+43+39+73+59+60+61",
    }
    pocket_dict_EV_A71_2Apro = {
        "catalytic": "19+39+110+124+125+126",
        "entry": "87+88+89+105+106+127",
        "rhs": "84+85+86+128+129",
        "rhs_prime": "82+83+98+100+131",
    }
    pocket_dict_ZIKV_NS2B_NS3pro = {
        "subP1": "151+161+129+160+150+130+131+132+135+134",
        "subP1_prime": "51+52+36+35+133",
        "subP2": "152+83+75+81+82+72",
        "subP3": "153+154+155",
    }
    pocket_dict_ZIKV_RdRppro = {"undecided": "1"}

    pocket_dict_DENV_NS2B_NS3pro = {
        "subP1": "135+151+161+159+129+150+130+132+131",
        "subP1_prime": "51+52+36+38+54",
        "subP2": "82+81+72+50+75+152",
        "subP3": "83+85+84+153+154+155+86",
    }
    pocket_dict_EV_A71_Capsid = {
        "hydrophic_trap": "155+135+137+24+190+179+177+188+133+233",
        "tube_region": "111+230+195+253+201+131+192+193+196",
        "entry_pore": "229+228+112+114+113+203+275+274+224",
    }

    pocket_dict_EV_D68_Capsid = {
        "hydrophic_trap": "147+119+121+182+171+168+180+117+220",
        "tube_region": "95+217+187+241+193+115+184+185+186",
        "entry_pore": "216+215+96+97+98+195+263+262+211",
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
    color_dict_N_protein = {
        "undecided": "grey",
    }
    color_dict_3Cpro = {
        "subP1": "orange",
        "subP1_prime": "yellow",
        "subP2": "blue",
        "subP3": "cyan",
        "subP4": "magenta",
    }

    color_dict_2Apro = {
        "catalytic": "orange",
        "entry": "yellow",
        "rhs": "blue",
        "rhs_prime": "magenta",
    }
    color_dict_NS2B_NS3pro = {
        "subP1": "yellow",
        "subP1_prime": "orange",
        "subP2": "blue",
        "subP3": "cyan",
    }
    color_dict_RdRppro = {
        "undecided": "grey",
    }
    color_dict_Capsid = {
        "hydrophic_trap": "orange",
        "tube_region": "blue",
        "entry_pore": "magenta",
    }
