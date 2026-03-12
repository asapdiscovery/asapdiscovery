import logging  # noqa: F401
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import xmltodict
import yaml
from asapdiscovery.data.backend.openeye import (
    combine_protein_ligand,
    oechem,
    save_openeye_pdb,
)
from asapdiscovery.data.metadata.resources import FINTSCORE_PARAMETERS
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.dataviz._gif_blocks import GIFBlockData
from asapdiscovery.spectrum.fitness import parse_fitness_json, target_has_fitness_data

logger = logging.getLogger(__name__)


def make_color_res_subpockets(protein, target) -> dict[str, str]:
    """
    Based on subpocket coloring, creates a dict where keys are colors, values are residue numbers.
    """
    # get a list of all residue numbers of the protein.
    protein_residues = [
        oechem.OEAtomGetResidue(atom).GetResidueNumber() for atom in protein.GetAtoms()
    ]

    # build a dict with all specified residue colorings.
    color_res_dict = {}
    binding_pocket_chainID = GIFBlockData.pocket_dict_chains_per_target[target]

    for subpocket, color in GIFBlockData.get_color_dict(target).items():
        subpocket_residues = GIFBlockData.get_pocket_dict(target)[subpocket].split("+")
        color_res_dict[color] = [
            f"{res}_{binding_pocket_chainID}" for res in subpocket_residues
        ]

    # set any non-specified residues to white.
    treated_res_nums = [
        f"{res}_{binding_pocket_chainID}"
        for sublist in color_res_dict.values()
        for res in sublist
    ]
    non_treated_res_nums = [
        f"{res}_{binding_pocket_chainID}"
        for res in set(protein_residues)
        if res not in treated_res_nums
    ]
    color_res_dict["white"] = non_treated_res_nums

    return color_res_dict


def make_color_res_fitness(protein, target) -> dict[str, str]:
    """
    Based on fitness coloring, creates a dict where keys are colors, values are residue numbers.
    """

    # get a list of all residue numbers of the protein.
    protein_residues = [
        oechem.OEAtomGetResidue(atom).GetResidueNumber() for atom in protein.GetAtoms()
    ]
    protein_chainIDs = [
        oechem.OEAtomGetResidue(atom).GetChainID() for atom in protein.GetAtoms()
    ]

    hex_color_codes = [
        "#ffffff",
        "#ff9e83",
        "#ff8a6c",
        "#ff7454",
        "#ff5c3d",
        "#ff3f25",
        "#ff0707",
    ]

    color_res_dict = {}
    json_data = parse_fitness_json(target)
    for res_num, chain in set(zip(protein_residues, protein_chainIDs)):
        try:
            # color residue white->red depending on fitness value.
            color_index_to_grab = json_data[f"{res_num}_{chain}"]
            try:
                color = hex_color_codes[color_index_to_grab]
            except IndexError:
                # insane residue that has tons of fit mutants; just assign the darkest red.
                color = hex_color_codes[-1]
            if color not in color_res_dict:
                color_res_dict[color] = [f"{res_num}_{chain}"]
            else:
                color_res_dict[color].append(f"{res_num}_{chain}")
        except KeyError:
            # fitness data is missing for this residue, color blue instead.
            color = "#642df0"
            if color not in color_res_dict:
                color_res_dict[color] = [f"{res_num}_{chain}"]
            else:
                color_res_dict[color].append(f"{res_num}_{chain}")

    return color_res_dict


def get_interaction_color(intn_type) -> str:
    """
    Generated using PLIP docs; colors match PyMol interaction colors. See
    https://github.com/pharmai/plip/blob/master/DOCUMENTATION.md
    converted RGB to HEX using https://www.rgbtohex.net/
    """
    if intn_type == "hydrophobic_interaction":
        return "#808080"
    elif intn_type == "hydrogen_bond":
        return "#0000FF"
    elif intn_type == "water_bridge":
        return "#BFBFFF"
    elif intn_type == "salt_bridge":
        return "#FFFF00"
    elif intn_type == "pi_stack":
        return "#00FF00"
    elif intn_type == "pi_cation_interaction":
        return "#FF8000"
    elif intn_type == "halogen_bond":
        return "#36FFBF"
    elif intn_type == "metal_complex":
        return "#8C4099"
    else:
        raise ValueError(f"Interaction type {intn_type} not recognized.")


def is_backbone_residue(protein, x, y, z) -> bool:
    """
    Given xyz coordinates, find the atom in the protein and return whether
    it is a backbone atom. This would be much easier if PLIP would return
    the atom idx of the protein, currently all we have are the coordinates.
    """
    # make a list with this protein's backbone atom indices. Could do higher up,
    # but this is very fast so ok to repeat.
    backbone_atoms = [at.GetIdx() for at in protein.GetAtoms(oechem.OEIsBackboneAtom())]

    # with oe, iterate over atoms until this one's found. then use oechem.OEIsBackboneAtom
    is_backbone = False
    for idx, res_coords in protein.GetCoords().items():
        # round to 3 because OE pointlessly extends the coordinates float.
        if (
            float(x) == round(res_coords[0], 3)
            and float(y) == round(res_coords[1], 3)
            and float(z) == round(res_coords[2], 3)
        ):
            is_backbone = True if idx in backbone_atoms else False

    if is_backbone:
        return True
    else:
        # this also catches pi-pi stack where protein coordinates are centered to a ring (e.g. Phe),
        # in which case the above coordinate matching doesn't find any atoms. pi-pi of this form
        # can never be on backbone anyway, so this works.
        return False


def get_interaction_fitness_color(plip_xml_dict, protein, target) -> str:
    """
    Get fitness color for a residue. If the interaction is with a backbone atom on
    the residue, color it green.
    """
    # first get the fitness color of the residue the interaction hits, this
    # can be white->red or blue if fitness data is missing.
    intn_color = None
    for fitness_color, res_ids in make_color_res_fitness(protein, target).items():
        if f"{plip_xml_dict['resnr']}_{plip_xml_dict['reschain']}" in res_ids:
            intn_color = fitness_color
            break

    # overwrite the interaction as green if it hits a backbone atom.
    if is_backbone_residue(
        protein,
        plip_xml_dict["protcoo"]["x"],
        plip_xml_dict["protcoo"]["y"],
        plip_xml_dict["protcoo"]["z"],
    ):
        intn_color = "#008000"

    return intn_color


def build_interaction_dict(
    plip_xml_dict, intn_counter, intn_type, color_method, protein, target
):
    """
    Parses a PLIP interaction dict and builds the dict key values needed for 3DMol.
    """
    k = f"{intn_counter}_{plip_xml_dict['restype']}{plip_xml_dict['resnr']}.{plip_xml_dict['reschain']}"

    if color_method == "fitness":
        intn_color = get_interaction_fitness_color(plip_xml_dict, protein, target)
    else:
        intn_color = get_interaction_color(intn_type)
    v = {
        "lig_at_x": plip_xml_dict["ligcoo"]["x"],
        "lig_at_y": plip_xml_dict["ligcoo"]["y"],
        "lig_at_z": plip_xml_dict["ligcoo"]["z"],
        "prot_at_x": plip_xml_dict["protcoo"]["x"],
        "prot_at_y": plip_xml_dict["protcoo"]["y"],
        "prot_at_z": plip_xml_dict["protcoo"]["z"],
        "type": intn_type,
        "color": intn_color,
    }
    return k, v


def get_interactions_plip(protein, pose, color_method, target) -> dict:
    """
    Get protein-ligand interactions according to PLIP.

    TODO:
    currently this uses a tmp PDB file, uses PLIP CLI (python package throws
    ```
    libc++abi: terminating with uncaught exception of type swig::stop_iteration
    Abort trap: 6
    ```
    ), then parses XML to get interactions. This is a bit convoluted, we could refactor
    this to use OE's InteractionHints instead? ProLIF struggles to detect incorrections
    because it's v sensitive to protonation. PLIP does protonation itself.
    """
    # create the complex PDB file.
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_pdb = os.path.join(tmpdirname, "tmp_complex.pdb")

        save_openeye_pdb(combine_protein_ligand(protein, pose), tmp_pdb)

        # run the PLIP CLI.
        subprocess.run(["plip", "-f", tmp_pdb, "-x", "-o", tmpdirname])

        # load the XML produced by PLIP that contains all the interaction data.
        intn_dict_xml = xmltodict.parse(
            ET.tostring(ET.parse(os.path.join(tmpdirname, "report.xml")).getroot())
        )

    intn_dict = {}
    intn_counter = 0
    # wrangle all interactions into a dict that can be read directly by 3DMol.

    # if there is only one site we get a dict, otherwise a list of dicts.
    sites = intn_dict_xml["report"]["bindingsite"]
    if isinstance(sites, dict):
        sites = [sites]

    for bs in sites:
        for _, data in bs["interactions"].items():
            if data:
                # we build keys for the dict to be unique, so no interactions are overwritten
                for intn_type, intn_data in data.items():
                    if isinstance(
                        intn_data, list
                    ):  # multiple interactions of this type
                        for intn_data_i in intn_data:
                            k, v = build_interaction_dict(
                                intn_data_i,
                                intn_counter,
                                intn_type,
                                color_method,
                                protein,
                                target,
                            )
                            intn_dict[k] = v
                            intn_counter += 1

                    elif isinstance(intn_data, dict):  # single interaction of this type
                        k, v = build_interaction_dict(
                            intn_data,
                            intn_counter,
                            intn_type,
                            color_method,
                            protein,
                            target,
                        )
                        intn_dict[k] = v
                        intn_counter += 1

    return intn_dict


# this should be placed around that area as well, but FINTscore should be added to docking scores by default
def compute_fint_score(
    protein: oechem.OEMol, pose: oechem.OEMol, target: TargetTags
) -> tuple[float, float]:
    """
    Compute the Fitness Interaction Score (FINTscore) given a dict with interactions generated by PLIP.

    Parameters
    ----------
    protein: oechem.OEMol
        Protein molecule
    pose: oechem.OEMol
        Pose molecule
    target: str
        Target name


    Returns
    ----------
    intn_score: float
        Score based purely on interactions, without penalties applied
    fint_score: float
        FINTscore which is computed as the `intn_score` with penalties applied
    """
    if not target_has_fitness_data(target):
        raise ValueError(
            f"Target {target} does not have fitness data, cannot compute FINTscore."
        )

    # read YAML file that contains settings for rewards/penalties of interaction types.
    fintscore_parameters = yaml.safe_load(Path(FINTSCORE_PARAMETERS).read_text())

    # set empty parameters to add to when iterating over interactions.
    penalty_multipliers = 1
    reward_multipliers = 1
    intn_score_bucket = []

    # iterate over each interaction that was found.
    intn_dict = get_interactions_plip(protein, pose, "fitness", target)

    for _, data in intn_dict.items():
        # if the interaction is with backbone, add a reward to the score.
        if data["color"] == "#008000":
            reward_multipliers += 1

        # if the interaction is with a residue that is shown to be able to mutate, add a penalty to the score.
        if data["color"] in fintscore_parameters.keys():
            penalty_multipliers += fintscore_parameters[data["color"]]

        # compute this interaction's score (set in metadata yaml).
        intn_score_bucket.append(fintscore_parameters[data["type"]])

    # simply compute the mean score, will end up being between 0.5 and 1.0, typically.
    # we need to take the mean because we don't want a compound with many interactions being favored by this score.
    intn_score = np.mean(intn_score_bucket)

    # now compute the FINTscore by applying the reward/penalty correction terms.
    # this follows FINT_{score} = INT_{score} * (REWARD*N_{backbone}) * PENALTY^{N_{mutable}},
    # where PENALTY <= 1.0 <= REWARD.
    fint_score = (
        intn_score
        * reward_multipliers
        * fintscore_parameters["backbone_reward_multiplier"]
        * fintscore_parameters["mutating_intn_penalty_multiplier"]
        ** penalty_multipliers
    )

    # finally, in some cases with lots of reward the FINTscore can shoot over 1.0; then just set as 1.0.
    if fint_score > 1.0:
        fint_score = 1.0

    return intn_score, fint_score
