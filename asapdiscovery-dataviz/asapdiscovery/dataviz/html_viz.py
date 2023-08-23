import logging
from pathlib import Path
from typing import Dict, Optional, Union  # noqa: F401

from asapdiscovery.data.fitness import parse_fitness_json
from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    load_openeye_pdb,
    load_openeye_sdf,
    oechem,
    oemol_to_pdb_string,
    openeye_perceive_residues,
    save_openeye_pdb,
)

from ._html_blocks import HTMLBlockData, make_core_html
from ._gif_blocks import GIFBlockData
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

        self._missed_residues = None
        if self.color_method == "fitness":
            self._missed_residues = self.make_fitness_bfactors()

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

    def make_color_res_subpockets(self) -> Dict:
        """
        Based on subpocket coloring, creates a dict where keys are colors, values are residue numbers.
        """

        # get a list of all residue numbers of the protein.
        protein_residues = [oechem.OEAtomGetResidue(atom).GetResidueNumber() \
                            for atom in self.protein.GetAtoms()]

        # build a dict with all specified residue colorings.
        color_res_dict = {}
        for subpocket, color in GIFBlockData.get_color_dict(self.target).items():
            subpocket_residues = GIFBlockData.get_pocket_dict(self.target)[subpocket].split("+")
            color_res_dict[color] = [ int(res) for res in subpocket_residues ]
        
        # set any non-specified residues to white.
        treated_res_nums = [ res for sublist in color_res_dict.values() for res in sublist ]
        non_treated_res_nums = [ res for res in set(protein_residues) if not res in treated_res_nums]
        color_res_dict["white"] = non_treated_res_nums
        
        return color_res_dict
    
    def make_color_res_fitness(self) -> Dict:
        """
        Based on fitness coloring, creates a dict where keys are colors, values are residue numbers.
        """
        # get a list of all residue numbers of the protein.
        protein_residues = [oechem.OEAtomGetResidue(atom).GetResidueNumber() \
                            for atom in self.protein.GetAtoms()]
        
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
                color = hex_color_codes[int(self.fitness_data[res_num]/10)]
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
        self.make_color_res_subpockets()
        self.make_color_res_fitness()
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
        missing_res = HTMLBlockData.get_missing_residues(self._missed_residues)
        orient_tail = HTMLBlockData.get_orient_tail(self.target)

        return colour + method + missing_res + orient_tail
