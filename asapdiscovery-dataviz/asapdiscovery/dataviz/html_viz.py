import logging  # noqa: F401
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Union  # noqa: F401
import pandas as pd 
pd.options.mode.chained_assignment = None
import logomaker
import matplotlib.pyplot as plt
import base64

import sys

import xmltodict
from airium import Airium
from asapdiscovery.data.fitness import parse_fitness_json, target_has_fitness_data, get_fitness_scores_bloom_by_target, _FITNESS_DATA_FIT_THRESHOLD
from asapdiscovery.data.metadata.resources import master_structures
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    oechem,
    oemol_to_pdb_string,
    oemol_to_sdf_string,
    openeye_perceive_residues,
    save_openeye_pdb,
)
from asapdiscovery.data.schema_v2.molfile import MolFileFactory
from asapdiscovery.modeling.modeling import split_openeye_mol, superpose_molecule
from asapdiscovery.data.postera.manifold_data_validation import (
    TargetVirusMap,
)
from ._gif_blocks import GIFBlockData
from ._html_blocks import HTMLBlockData
from .viz_targets import VizTargets

logger = logging.getLogger(__name__)


class HTMLVisualizer:
    """
    Class for generating HTML visualizations of poses.
    """

    allowed_targets = VizTargets.get_allowed_targets()

    # TODO: replace input with a schema rather than paths.
    def __init__(
        self,
        poses: list[Path],
        output_paths: list[Path],
        target: str,
        protein: Path,
        color_method: str = "subpockets",
        align=True,
        debug: bool = False,
    ):
        """
        Parameters
        ----------
        poses : List[Path]
            List of poses to visualize, in SDF format.
        output_paths : List[Path]
            List of paths to write the visualizations to.
        target : str
            Target to visualize poses for. Must be one of the allowed targets in VizTargets.
        protein : Path
            Path to protein PDB file.
        color_method : str
            Protein surface coloring method. Can be either by `subpockets` or `fitness`
        align : bool
            Whether or not to align the protein (and poses) to the master structure of the target. Redundant if already docked.
        """
        if not len(poses) == len(output_paths):
            raise ValueError("Number of poses and paths must be equal.")

        if target not in self.allowed_targets:
            raise ValueError(
                f"Target {target} invalid, must be one of: {self.allowed_targets}"
            )
        self.target = target
        self.reference_target = load_openeye_pdb(master_structures[self.target])
        self.align = align

        self.debug = debug

        self.color_method = color_method
        if self.color_method == "subpockets":
            pass
        elif self.color_method == "fitness":
            if not target_has_fitness_data(self.target):
                raise NotImplementedError(
                    f"No viral fitness data available for {self.target}: set `color_method` to `subpockets`."
                )
            self.fitness_data = parse_fitness_json(self.target)
            self.fitness_data_logoplots = get_fitness_scores_bloom_by_target(self.target)
        else:
            raise ValueError(
                "variable `color_method` must be either of ['subpockets', 'fitness']"
            )

        self.poses = []
        self.output_paths = []
        # make sure all paths exist, otherwise skip
        for pose, path in zip(poses, output_paths):
            if pose:
                if isinstance(pose, oechem.OEMolBase):
                    mol = pose.CreateCopy()
                else:
                    mol = MolFileFactory.from_file(
                        str(pose)
                    ).ligands  # in this way we allow multiple ligands per protein, e.g. for viewing fragments
                    mol = [l.to_oemol() for l in mol]
                    for m in mol:
                        oechem.OESuppressHydrogens(
                            m, True, True
                        )  # retain polar hydrogens and hydrogens on chiral centers
                self.poses.append(mol)

                self.output_paths.append(path)
            else:
                pass

        if isinstance(protein, oechem.OEMolBase):
            self.protein = protein.CreateCopy()
        else:
            if not protein.exists():
                raise ValueError(f"Protein {protein} does not exist.")
            self.protein = openeye_perceive_residues(
                load_openeye_pdb(protein), preserve_all=True
            )
        if target == "EV-A71-Capsid":
            # because capsid has an encapsulated ligand, we need to Z-clip.
            self.slab = "viewer.setSlab(-11, 50)\n"
        else:
            self.slab = ""

    @staticmethod
    def write_html(html, path):
        """
        Write HTML to a file.

        Parameters
        ----------
        html : str
            HTML to write.
        path : Path
            Path to write HTML to.
        """
        with open(path, "w") as f:
            f.write(html)

    def get_color_dict(self) -> dict:
        """
        Depending on color type, return a dict that contains residues per color.
        """
        if self.color_method == "subpockets":
            return self.make_color_res_subpockets()
        elif self.color_method == "fitness":
            return self.make_color_res_fitness()

    def make_color_res_subpockets(self) -> dict:
        """
        Based on subpocket coloring, creates a dict where keys are colors, values are residue numbers.
        """
        # get a list of all residue numbers of the protein.
        protein_residues = [
            oechem.OEAtomGetResidue(atom).GetResidueNumber()
            for atom in self.protein.GetAtoms()
        ]

        # build a dict with all specified residue colorings.
        color_res_dict = {}
        for subpocket, color in GIFBlockData.get_color_dict(self.target).items():
            subpocket_residues = GIFBlockData.get_pocket_dict(self.target)[
                subpocket
            ].split("+")
            color_res_dict[color] = [int(res) for res in subpocket_residues]

        # set any non-specified residues to white.
        treated_res_nums = [
            res for sublist in color_res_dict.values() for res in sublist
        ]
        non_treated_res_nums = [
            res for res in set(protein_residues) if res not in treated_res_nums
        ]
        color_res_dict["white"] = non_treated_res_nums

        return color_res_dict

    def make_color_res_fitness(self) -> dict:
        """
        Based on fitness coloring, creates a dict where keys are colors, values are residue numbers.
        """

        # get a list of all residue numbers of the protein.
        protein_residues = [
            oechem.OEAtomGetResidue(atom).GetResidueNumber()
            for atom in self.protein.GetAtoms()
        ]
        protein_chainIDs = [
            oechem.OEAtomGetResidue(atom).GetChainID()
            for atom in self.protein.GetAtoms()
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
        for res_num, chain in set(zip(protein_residues, protein_chainIDs)):

            try:
                # color residue white->red depending on fitness value.
                color_index_to_grab = self.fitness_data[f"{res_num}_{chain}"]
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

    @staticmethod
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

    def is_backbone_residue(self, x, y, z) -> bool:
        """
        Given xyz coordinates, find the atom in the protein and return whether
        it is a backbone atom. This would be much easier if PLIP would return
        the atom idx of the protein, currently all we have are the coordinates.
        """
        # make a list with this protein's backbone atom indices. Could do higher up,
        # but this is very fast so ok to repeat.
        backbone_atoms = [
            at.GetIdx() for at in self.protein.GetAtoms(oechem.OEIsBackboneAtom())
        ]

        # with oe, iterate over atoms until this one's found. then use oechem.OEIsBackboneAtom
        is_backbone = False
        for idx, res_coords in self.protein.GetCoords().items():
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

    def get_interaction_fitness_color(self, plip_xml_dict) -> str:
        """
        Get fitness color for a residue. If the interaction is with a backbone atom on
        the residue, color it green.
        """
        # first get the fitness color of the residue the interaction hits, this
        # can be white->red or blue if fitness data is missing.
        intn_color = None
        for fitness_color, res_ids in self.make_color_res_fitness().items():
            res_nums = [ int(id.split("_")[0]) for id in res_ids]
            if int(plip_xml_dict["resnr"]) in res_nums:
                intn_color = fitness_color
                break

        # overwrite the interaction as green if it hits a backbone atom.
        if self.is_backbone_residue(
            plip_xml_dict["protcoo"]["x"],
            plip_xml_dict["protcoo"]["y"],
            plip_xml_dict["protcoo"]["z"],
        ):
            intn_color = "#008000"

        return intn_color

    def build_interaction_dict(self, plip_xml_dict, intn_counter, intn_type):
        """
        Parses a PLIP interaction dict and builds the dict key values needed for 3DMol.
        """
        k = f"{intn_counter}_{plip_xml_dict['restype']}{plip_xml_dict['resnr']}.{plip_xml_dict['reschain']}"

        if self.color_method == "fitness":
            intn_color = self.get_interaction_fitness_color(plip_xml_dict)
        else:
            intn_color = self.get_interaction_color(intn_type)
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

    @staticmethod
    def get_interactions_plip(self, pose) -> dict:
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

            save_openeye_pdb(combine_protein_ligand(self.protein, pose), tmp_pdb)

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
                                k, v = self.build_interaction_dict(
                                    intn_data_i, intn_counter, intn_type
                                )
                                intn_dict[k] = v
                                intn_counter += 1

                        elif isinstance(
                            intn_data, dict
                        ):  # single interaction of this type
                            k, v = self.build_interaction_dict(
                                intn_data, intn_counter, intn_type
                            )
                            intn_dict[k] = v
                            intn_counter += 1

        return intn_dict

    def get_html_airium(self, pose):
        """
        Get HTML for visualizing a single pose. This uses Airium which is a handy tool to write
        HTML using python. We can't do f-string because of all the JS curly brackets, need to do '+' instead.
        """
        a = Airium()

        # first check if we need to align the protein and ligand. This already happens during docking, but not
        # during pose_to_viz.py.
        if self.align:
            # merge
            complex = self.protein
            for pos in pose:
                complex = combine_protein_ligand(complex, pos)

            # align complex to master structure
            complex_aligned, _ = superpose_molecule(
                self.reference_target,
                complex,
            )

            # get pose and protein back
            opts = oechem.OESplitMolComplexOptions()
            opts.SetProteinFilter(
                oechem.OEOrRoleSet(
                    oechem.OEMolComplexFilterFactory(
                        oechem.OEMolComplexFilterCategory_Protein
                    ),
                    oechem.OEMolComplexFilterFactory(
                        oechem.OEMolComplexFilterCategory_Peptide
                    ),
                )
            )
            # pretend that the binding site is huge so we make sure that we include all ligands.
            opts.SetMaxBindingSiteDist(1000)

            pose = oechem.OEGraphMol()
            self.protein = oechem.OEGraphMol()
            oechem.OESplitMolComplex(
                pose,
                self.protein,
                oechem.OEGraphMol(),
                oechem.OEGraphMol(),
                complex_aligned,
                opts,
            )

        # now prep the coloring function.
        surface_coloring = self.get_color_dict()

        residue_coloring_function_js = ""
        start = True
        for color, residues in surface_coloring.items():
            residues = [f"'{res}'" for res in residues] # need to wrap the string in quotes *within* the JS code
            if start:
                residue_coloring_function_js += (
                    "if (["
                    + ",".join(residues)
                    + "].includes(atom.resi+'_'+atom.chain)){ \n return '"
                    + color
                    + "' \n "
                )
                start = False
            else:
                residue_coloring_function_js += (
                    "} else if (["
                    + ",".join(residues)
                    + "].includes(atom.resi+'_'+atom.chain)){ \n return '"
                    + color
                    + "' \n "
                )

        # start writing the HTML doc.
        a("<!DOCTYPE HTML>")
        with a.html(lang="en"):
            with a.head():
                a.meta(charset="utf-8")
                a.script(
                    crossorigin="anonymous",
                    integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN",
                    src="https://code.jquery.com/jquery-3.2.1.slim.min.js",
                )
                a.script(
                    crossorigin="anonymous",
                    integrity="sha384-vFJXuSJphROIrBnz7yo7oB41mKfc8JzQZiCq4NCceLEaO4IHwicKwpJf9c9IpFgh",
                    src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.3/umd/popper.min.js",
                )
                a.script(
                    crossorigin="anonymous",
                    integrity="sha384-alpBpkh1PFOepccYVYDB4do5UnbKysX5WZXm3XxPqe5iKTfUKjNkCk9SaVuEZflJ",
                    src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/js/bootstrap.min.js",
                )
                a.link(
                    crossorigin="anonymous",
                    href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta.2/css/bootstrap.min.css",
                    integrity="sha384-PsH8R72JQ3SOdhVi3uxftmaW6Vc51MKb0q5P2rRUpPvrszuE4W1povHYgTpBfshb",
                    rel="stylesheet",
                )
                a.script(src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js")
                a.script(src="https://d3js.org/d3.v5.min.js")
                with a.style():
                    a('/* Dropdown Button */\n      .dropbtn {\n        background-color: #04AA6D;\n        color: white;\n        padding: 16px;\n        font-size: 16px;\n        border: none;\n        border-radius: 5;\n      }\n\n      /* The container <div> - needed to position the dropdown content */\n      .dropdown {\n        position: absolute;\n        display: inline-block;\n        left: 1%;\n        top: 1%;\n      }\n      .dropdown_ctcs {\n        position: absolute;\n        top: 7%;\n        left: 1%;\n        display: inline-block;\n      }\n\n  .dropdown_lgplts {\n        position: absolute;\n        top: 13%;\n        left: 1%;\n        display: inline-block;\n      }\n\n    /* Dropdown Content (Hidden by Default) */\n      .dropdown-content {\n        display: none;\n        position: relative;\n        background-color: #f1f1f1;\n        min-width: 160px;\n        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);\n        z-index: 1;\n      }\n\n      /* Links inside the dropdown */\n      .dropdown-content a {\n        color: black;\n        padding: 12px 16px;\n        text-decoration: none;\n        display: block;\n        cursor: default;\n      }\n                                                              \n      /* Show the dropdown menu on hover */\n      .dropdown:hover .dropdown-content {display: block;}\n      .dropdown_ctcs:hover .dropdown-content {display: block;}\n   .dropdown_lgplts:hover .dropdown-content {display: block;}\n   \n      /* Change the background color of the dropdown button when the dropdown content is shown */\n      .dropdown:hover .dropbtn {background-color: #3e8e41;}\n    \n\n      .viewerbox {\n        position: absolute;\n        width: 200px;\n        height: 100px;\n        padding: 10px;\n      }\n\n      .logoplotbox_unfit {\n        position: absolute;\n        top: 35%;\n        right:1%;\n        border: 5px solid black;\n      }\n      .logoplotbox_fit {\n        position: absolute;\n        top: 35%;\n        left:1%;\n        border: 5px solid black;\n      }')
                a("<!-- wrap the main JS block to please the frontend gods -->")
                with a.div(klass="box"):
                    a.div(id="gldiv", style="width: 100vw; height: 100vh;")

                a("<!-- show the top dropdown (surfaces) -->")
                with a.div(klass="dropdown"):
                    a.button(klass="dropbtn", _t="Key (Surfaces)")
                    with a.div(klass="dropdown-content", style="text-align: center"):
                        a.a(
                            href="#",
                            _t="Protein residue surfaces are colored by mutability:",
                        )
                    with a.div(klass="dropdown-content"):
                        a.a(href="#", _t="⚪ : No fit mutants for residue")
                        a.a(
                            href="#",
                            _t="🔴 : Fit mutants for residue (increasing 🔴 intensity with n fit mutants)",
                        )
                        a.a(href="#", _t="🟣 : No data for residue")

                a("<!-- show the bottom dropdown (contacts) -->")
                with a.div(klass="dropdown_ctcs"):
                    a.button(klass="dropbtn", _t="Key (Contacts)")
                    with a.div(klass="dropdown-content", style="text-align: center"):
                        a.a(
                            href="#",
                            _t="Ligand-protein contacts are shown as dashed lines colored by:",
                        )
                    with a.div(klass="dropdown-content"):
                        a.a(href="#", _t="⬜ : No fit mutants for contacted residue")
                        a.a(href="#", _t="🟩 : Ligand contacts residue backbone")
                        a.a(
                            href="#",
                            _t="🟥 : Fit mutants for contacted residue (increasing 🟥 intensity with n fit mutants)",
                        )
                        a.a(href="#", _t="🟪 : No data for contacted residue")

                a("<!-- show the bottom dropdown (logoplots) -->")
                with a.div(klass="dropdown_lgplts"):
                    a.button(klass="dropbtn", _t="Key (Logo Plots)")
                    with a.div(klass="dropdown-content", style="text-align: center"):
                        a.a(
                            href="#",
                            _t="Fitness logo plots are shown on hover of residue atoms:",
                        )
                    with a.div(klass="dropdown-content"):
                        a.a(href="#", _t="Left panel : Wildtype and fit mutant residues<br /> Right panel : Non-fit mutant residues")
                        a.a(href="#", _t="The height of each residue letter corresponds to its relative fitness:<br /> in the fit residues panel (left) residues are ordered by increasing fitness (from bottom to top);<br /> in the non-fit residues panel (right), residues are ordered by decreasing fitness (top to bottom).<br /> An asterisk (*) denotes the stop codon mutation - this negative control should be non-fit.")


                a('<!-- show logoplots per residue on hover -->')
                a('<!-- bake in the base64 divs of all the residues. -->')
                for resi, _ in self.fitness_data.items():
                    resnum, chain = resi.split("_")
                    # get the base64 for this residue in this chain. 
                    for fit_type, base64 in self.make_logoplot_input(resi).items():
                        
                        with a.div(klass=f'logoplotbox_{fit_type}', id=f'{fit_type}DIV_{resnum}_{chain}', style='display:none'):
                            # add the base64 string while making some corrections. 
                            a.img(alt=f'{fit_type} residue logoplot', src=str(base64).replace("b'", "data:image/png;base64,").replace("'", ""))
                        # with a.div(klass='logoplotbox_fit', id=f'fitDIV_{resnum}_{chain}', style='display:none'):
                        #     a.img(alt='fit residue logoplot', src=str(base64_dict["fit"]).replace("b'", "data:image/png;base64,").replace("'", ""))
                
            
            with a.script():
                # function to show/hide the logoplots
                a('function showLogoPlots(resi, chain) {\n        var x = document.getElementById("fitDIV_"+resi+"_"+chain);\n        var y = document.getElementById("unfitDIV_"+resi+"_"+chain);\n        x.style.display = "block";\n        y.style.display = "block";\n\n      }\n      function hideLogoPlots(resi, chain) {\n        var x = document.getElementById("fitDIV_"+resi+"_"+chain);\n        var y = document.getElementById("unfitDIV_"+resi+"_"+chain);\n        x.style.display = "none";\n        y.style.display = "none";\n      }')
                
                # function to show 3DMol viewer
                a(
                    'var viewer=$3Dmol.createViewer($("#gldiv"));\n \
                    var prot_pdb = `    '
                    + oemol_to_pdb_string(self.protein)
                    + "\n \
                    \n \
                    `;\n \
                    var lig_sdf =`  "
                    + oemol_to_sdf_string(pose)
                    + '\n \
                    `;       \n \
                        //////////////// set up system\n \
                        viewer.addModel(prot_pdb, "pdb") \n \
                        // set protein sticks and surface\n \
                        viewer.setStyle({model: 0}, {stick: {colorscheme: "whiteCarbon", radius:0.15}});\n \
                        // define a coloring function based on our residue ranges. We can\'t call .addSurface separate times because the surfaces won\'t be merged nicely. \n \
                        var colorAsSnake = function(atom) { \
                        '
                    + residue_coloring_function_js
                    + " \
                                        }}; \
                        viewer.addSurface(\"MS\", {colorfunc: colorAsSnake, opacity: 0.9}) \n \
                    \n \
                        viewer.setStyle({bonds: 0}, {sphere:{radius:0.5}}); //water molecules\n \
                    \n \
                        viewer.addModel(lig_sdf, \"sdf\")   \n \
                        // set ligand sticks\n \
                        viewer.setStyle({model: -1}, {stick: {colorscheme: \"pinkCarbon\"}});\n \
                    \n \
                        ////////////////// enable show residue number on hover\n \
                        viewer.setHoverable({}, true,\n \
                        function (atom, viewer, event, container) {\n \
                            console.log('hover', atom);\n \
                            console.log('view:', viewer.getView()); // to get view for system\n \
                            if (!atom.label) {\n \
                                if (atom.chain === undefined){ \
                                    display_str = 'LIGAND'; \
                                    } else { \
                                    display_str = atom.chain + ': ' +  atom.resn + atom.resi; \
                                    showLogoPlots(atom.resi, atom.chain); \
                                } \
                                atom.label = viewer.addLabel(display_str, { position: atom, backgroundColor: 'mintcream', fontColor: 'black' }); \
                            }\n \
                        },\n \
                        function (atom) {\n \
                            if (atom.label) {\n \
                                viewer.removeLabel(atom.label);\n \
                                delete atom.label;\n \
                                hideLogoPlots(atom.resi, atom.chain); \
                            }\n \
                        }\n \
                        );\n \
                        viewer.setHoverDuration(100); // makes resn popup instant on hover\n \
                    \n \
                        //////////////// add protein-ligand interactions\n \
                        var intn_dict = "
                    + str(self.get_interactions_plip(self, pose))
                    + '\n \
                        for (const [_, intn] of Object.entries(intn_dict)) {\n \
                            viewer.addCylinder({start:{x:parseFloat(intn["lig_at_x"]),y:parseFloat(intn["lig_at_y"]),z:parseFloat(intn["lig_at_z"])},\n \
                                                    end:{x:parseFloat(intn["prot_at_x"]),y:parseFloat(intn["prot_at_y"]),z:parseFloat(intn["prot_at_z"])},\n \
                                                    radius:0.1,\n \
                                                    dashed:true,\n \
                                                    fromCap:2,\n \
                                                    toCap:2,\n \
                                                    color:intn["color"]},\n \
                                                    );\n \
                        }\n \
                    \n \
                        ////////////////// set the view correctly\n \
                        viewer.setBackgroundColor(0xffffffff);\n \
                        viewer.setView(\n \
                        '
                    + HTMLBlockData.get_orient()
                    + " \
                        )\n\
                        viewer.setZoomLimits(1,250) // prevent infinite zooming\n"
                    + self.slab
                    + " viewer.render();"
                )

        return str(a)

    def write_pose_visualizations(self):
        """
        Write HTML visualisations for all poses.
        """
        output_paths = []
        for pose, path in zip(self.poses, self.output_paths):
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            outpath = self.write_pose_visualization(pose, path)
            output_paths.append(outpath)
        return output_paths

    def write_pose_visualization(self, pose, path):
        """
        Write HTML visualisation for a single pose.
        """
        html = self.get_html_airium(pose)
        self.write_html(html, path)
        return path

    def make_poses_html(self):
        html_renders = []
        for pose, path in zip(self.poses, self.output_paths):
            html = self.get_html_airium(pose)
            html_renders.append(html)
        return html_renders

    def make_logoplot_input(self, resi) -> dict:
        """
        given a residue number with underscored chain ID, get data for the fitness of all mutants for the residue. Use
        LogoMaker to create a logoplot for both the fit and unfit mutants, return the base64
        string of the image.
        """

        # get just the fitness data for the queried residue index, at the right chain.
        resi, chain = resi.split("_")
        
        site_df_resi = self.fitness_data_logoplots[self.fitness_data_logoplots["site"] == int(resi)]
        site_df = site_df_resi[site_df_resi["chain"] == chain]
        
        # add the fitness threshold to normalize so that fit mutants end up in the left-hand logoplot.
        site_df["fitness"] = site_df["fitness"] + abs(_FITNESS_DATA_FIT_THRESHOLD[TargetVirusMap[self.target]])

        # split the mutant data into fit/unfit. 
        site_df_fit = site_df[site_df["fitness"] > 0 ]
        site_df_unfit = site_df[site_df["fitness"] < 0 ]

        logoplot_base64s_dict = {}
        with tempfile.TemporaryDirectory() as tmpdirname:
            for fit_type, fitness_df in zip(["fit", "unfit"], [site_df_fit, site_df_unfit]):
                # pivot table to make into LogoMaker format
                logoplot_df = pd.DataFrame([fitness_df["fitness"].values], columns=fitness_df["mutant"])

                # create Logo object
                logomaker.Logo(logoplot_df,
                                    shade_below=.5,
                                    fade_below=.5,
                                    font_name='Sans Serif',
                                    figsize=(3,10),
                                    color_scheme="dmslogo_funcgroup",
                                    flip_below=False,
                                    show_spines=True)
                plt.xticks([])
                plt.yticks([])

                # we could get base64 from buffer, but easier to write as tmp and read back as bas64.
                plt.savefig(f"{tmpdirname}/logoplot.png", bbox_inches='tight', pad_inches=0, dpi=50)
                plt.close() # prevent matplotlib from freaking out due to large volume of figures.
                with open(f"{tmpdirname}/logoplot.png", "rb") as f:
                    logoplot_base64s_dict[fit_type] = base64.b64encode(f.read())

        return logoplot_base64s_dict


