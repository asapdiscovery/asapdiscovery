import logging
from pathlib import Path
from typing import Dict, Optional, Union  # noqa: F401

from airium import Airium
from asapdiscovery.data.fitness import parse_fitness_json
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    load_openeye_sdf,
    oechem,
    oemol_to_pdb_string,
    oemol_to_sdf_string,
    openeye_perceive_residues,
    save_openeye_pdb,
)

from ._gif_blocks import GIFBlockData
from ._html_blocks import HTMLBlockData, make_core_html
from .viz_targets import VizTargets


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
        logger: FileLogger = None,
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
        logger : FileLogger
            Logger to use

        """
        if not len(poses) == len(output_paths):
            raise ValueError("Number of poses and paths must be equal.")

        if target not in self.allowed_targets:
            raise ValueError(f"Target must be one of: {self.allowed_targets}")
        self.target = target

        # init loggers
        if logger is None:
            self.logger = FileLogger(
                "html_visualizer_log.txt", "./", stdout=True, level=logging.INFO
            ).getLogger()
        else:
            self.logger = logger
        self.debug = debug

        self.color_method = color_method
        if self.color_method == "subpockets":
            self.logger.info("Mapping interactive view by subpocket dict")
        elif self.color_method == "fitness":
            if self.target == "MERS-CoV-Mpro":
                raise NotImplementedError(
                    "No viral fitness data available for MERS-CoV-Mpro: set `color_method` to `subpockets`."
                )
            self.logger.info(
                "Mapping interactive view by fitness (visualised with b-factor)"
            )
            self.fitness_data = parse_fitness_json(self.target)
        else:
            raise ValueError(
                "variable `color_method` must be either of ['subpockets', 'fitness']"
            )

        self.logger.info(f"Visualising poses for {self.target}")

        self.poses = []
        self.output_paths = []
        # make sure all paths exist, otherwise skip
        for pose, path in zip(poses, output_paths):
            if pose and Path(pose).exists():
                self.poses.append(load_openeye_sdf(str(pose)))
                self.output_paths.append(path)
            else:
                self.logger.warning(f"Pose {pose} does not exist, skipping.")

        if not protein.exists():
            raise ValueError(f"Protein {protein} does not exist.")

        self.protein = openeye_perceive_residues(
            load_openeye_pdb(str(protein)), preserve_all=True
        )

        self.logger.debug(
            f"Writing HTML visualisations for {len(self.output_paths)} ligands"
        )

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
            res for res in set(protein_residues) if not res in treated_res_nums
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

        hex_color_codes = [
            "#ffffff",
            "#ffece5",
            "#ffd9cc",
            "#ffc6b3",
            "#ffb29b",
            "#ff9e83",
            "#ff8a6c",
            "#ff7454",
            "#ff5c3d",
            "#ff3f25",
            "#ff0707",
        ]
        color_res_dict = {}
        for res_num in set(protein_residues):
            try:
                # color residue white->red depending on fitness value.
                color = hex_color_codes[int(self.fitness_data[res_num] / 10)]
                if not color in color_res_dict:
                    color_res_dict[color] = [res_num]
                else:
                    color_res_dict[color].append(res_num)
            except KeyError:
                # fitness data is missing for this residue, color blue instead.
                color = "#642df0"
                if not color in color_res_dict:
                    color_res_dict[color] = [res_num]
                else:
                    color_res_dict[color].append(res_num)

        return color_res_dict

    def get_html_airium(self, pose):
        """
        Get HTML for visualizing a single pose. This uses Airium which is a handy tool to write
        HTML using python. We can't do f-string because of all the JS curly brackets, need to do '+' instead.
        """
        a = Airium()

        # first prep the coloring function.
        surface_coloring = self.get_color_dict()
        residue_coloring_function_js = ""
        start = True
        for color, residues in surface_coloring.items():
            residues = [str(res) for res in residues]
            if start:
                residue_coloring_function_js += (
                    "if (["
                    + ",".join(residues)
                    + "].includes(atom.resi)){ \n return '"
                    + color
                    + "' \n "
                )
                start = False
            else:
                residue_coloring_function_js += (
                    "} else if (["
                    + ",".join(residues)
                    + "].includes(atom.resi)){ \n return '"
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
            with a.body():
                a.div(
                    id="gldiv", style="width: 100vw; height: 100vh; position: relative;"
                )
                with a.script():
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
                            viewer.addSurface(\"VDW\", {colorfunc: colorAsSnake, opacity: 0.9}) \n \
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
                                    atom.label = viewer.addLabel(atom.resn + atom.resi, { position: atom, backgroundColor: 'mintcream', fontColor: 'black' });\n \
                                    // if fitness view, can we show all possible residues it can mutate into with decent fitness?\n \
                                }\n \
                            },\n \
                            function (atom) {\n \
                                console.log('unhover', atom);\n \
                                if (atom.label) {\n \
                                    viewer.removeLabel(atom.label);\n \
                                    delete atom.label;\n \
                                }\n \
                            }\n \
                            );\n \
                            viewer.setHoverDuration(100); // makes resn popup instant on hover\n \
                        \n \
                            //////////////// add protein-ligand interactions\n \
                            var intn_dict = {'3_ILE23.A': {'lig_at_x': 6.1168, 'lig_at_y': -15.1724, 'lig_at_z': 15.0378, 'prot_at_x': 7.836, 'prot_at_y': -14.648, 'prot_at_z': 12.562, 'type': 'HBAcceptor', 'color': 'yellow'}, '4_VAL49.A': {'lig_at_x': 4.6893, 'lig_at_y': -13.8543, 'lig_at_z': 22.5618, 'prot_at_x': 7.458, 'prot_at_y': -13.695, 'prot_at_z': 21.342, 'type': 'HBAcceptor', 'color': 'yellow'}, '5_ILE131.A': {'lig_at_x': 2.7582, 'lig_at_y': -15.0623, 'lig_at_z': 23.3433, 'prot_at_x': 0.121, 'prot_at_y': -15.462, 'prot_at_z': 24.94, 'type': 'HBAcceptor', 'color': 'yellow'}, '8_PHE156.A': {'lig_at_x': 6.86024, 'lig_at_y': -16.45794, 'lig_at_z': 17.0304, 'prot_at_x': 6.938333333333333, 'prot_at_y': -20.451999999999998, 'prot_at_z': 14.069999999999999, 'type': 'PiStacking', 'color': 'purple'}};\n \
                        \n \
                            for (const [_, intn] of Object.entries(intn_dict)) {\n \
                                viewer.addCylinder({start:{x:intn[\"lig_at_x\"],y:intn[\"lig_at_y\"],z:intn[\"lig_at_z\"]},\n \
                                                        end:{x:intn[\"prot_at_x\"],y:intn[\"prot_at_y\"],z:intn[\"prot_at_z\"]},\n \
                                                        radius:0.1,\n \
                                                        dashed:true,\n \
                                                        fromCap:2,\n \
                                                        toCap:2,\n \
                                                        color:intn[\"color\"]},\n \
                                                        );\n \
                            }\n \
                        \n \
                            ////////////////// set the view correctly\n \
                            viewer.setBackgroundColor(0xffffffff);\n \
                            viewer.setView(\n \
                            "
                        + HTMLBlockData.get_orient(self.target)
                        + " \
                            )\n \
                            viewer.setZoomLimits(1,250) // prevent infinite zooming\n \
                            viewer.render();"
                    )

        print(a)

    def make_fitness_bfactors(self) -> set[int]:
        """
        Given a dict of fitness values, swap out the b-factors in the protein.
        """

        self.logger.info("Swapping b-factor with fitness score.")

        # this loop looks a bit strange but OEResidue state is not saved without a loop over atoms
        missed_res = set()
        for atom in self.protein.GetAtoms():
            thisRes = oechem.OEAtomGetResidue(atom)
            res_num = thisRes.GetResidueNumber()
            thisRes.SetBFactor(
                0.0
            )  # reset b-factor to 0 for all residues first, so that missing ones can have blue overlaid nicely.
            try:
                thisRes.SetBFactor(self.fitness_data[res_num])
            except KeyError:
                missed_res.add(res_num)
            oechem.OEAtomSetResidue(atom, thisRes)  # store updated residue

        if self.debug:
            self.logger.info(
                f"Missed {len(missed_res)} residues when mapping fitness data."
            )
            self.logger.info(f"Missed residues: {missed_res}")
            self.logger.info("Writing protein with fitness b-factors to file.")
            save_openeye_pdb(self.protein, "protein_fitness.pdb")

        return missed_res

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
        html = self.get_html(pose)
        self.get_html_airium(pose)
        self.write_html(html, path)
        return path

    def get_html(self, pose):
        """
        Get HTML for visualizing a single pose.
        """
        return self.get_html_body(pose) + self.get_html_footer()

    def get_html_body(self, pose):
        """
        Get HTML body for pose visualization
        """
        mol = combine_protein_ligand(self.protein, pose)
        oechem.OESuppressHydrogens(mol, True)  # remove hydrogens retaining polar ones
        joint_pdb = oemol_to_pdb_string(mol)

        html_body = make_core_html(joint_pdb)
        return html_body

    def get_html_footer(self):
        """
        Get HTML footer for pose visualization
        """

        colour = HTMLBlockData.get_pocket_color(self.target)

        method = HTMLBlockData.get_color_method(self.color_method)
        orient_tail = HTMLBlockData.get_orient_tail(self.target)

        return colour + method + orient_tail
