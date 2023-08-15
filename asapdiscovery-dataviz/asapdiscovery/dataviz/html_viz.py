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
        joint_pdb = oemol_to_pdb_string(combine_protein_ligand(self.protein, pose))

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
