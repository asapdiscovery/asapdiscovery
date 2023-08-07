import logging
import os
from pathlib import Path
from typing import List, Optional, Union  # noqa: F401

from asapdiscovery.data.logging import FileLogger
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from rdkit import Chem

from ._html_blocks import HTMLBlockData, make_core_html
from .viz_targets import VizTargets


def _load_first_molecule(file_path: Union[Path, str]):
    mols = Chem.SDMolSupplier(str(file_path))
    return mols[0]


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
        color_method: str,
        fitness_data: bool = False,
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
            Protein surface coloring method. Can be either by `subpockets` or `bfactor`
        fitness_data: dict
            dict of k,v where k is residue number and v is a fitness value between 0-100.
        logger : FileLogger
            Logger to use

        """
        if not len(poses) == len(output_paths):
            raise ValueError("Number of poses and paths must be equal.")

        # init loggers
        if logger is None:
            self.logger = FileLogger(
                "html_visualizer_log.txt", "./", stdout=True, level=logging.INFO
            ).getLogger()
        else:
            self.logger = logger

        self.color_method = color_method
        if self.color_method == "subpockets":
            self.logger.info(f"Mapping interactive view by subpocket dict")
        elif self.color_method == "bfactor":
            self.logger.info(f"Mapping interactive view by fitness (b-factor bypass)")
            self.fitness_data = fitness_data
        else:
            raise ValueError(
                "variable `color_method` must be either of ['subpockets', 'bfactor']"
            )

        if target not in self.allowed_targets:
            raise ValueError(f"Target must be one of: {self.allowed_targets}")
        self.target = target
        self.logger.info(f"Visualising poses for {self.target}")

        self.poses = []
        self.output_paths = []
        # make sure all paths exist, otherwise skip
        for pose, path in zip(poses, output_paths):
            if pose and Path(pose).exists():
                self.poses.append(_load_first_molecule(pose))
                self.output_paths.append(path)
            else:
                self.logger.warning(f"Pose {pose} does not exist, skipping.")

        if not protein.exists():
            raise ValueError(f"Protein {protein} does not exist.")
        if self.color_method == "subpockets":
            self.protein = Chem.MolFromPDBFile(str(protein))
        elif self.color_method == "bfactor":
            # first need to swap the protein's b-factor with fitness scores. Could do this with str.split() but gets a bit complicated.
            parser = PDBParser()
            protein_biopython = parser.get_structure("protein", str(protein))
            self.logger.warning(f"Swapping b-factor with fitness score.")
            for res in protein_biopython.get_residues():
                res_number = res.get_full_id()[3][1]  # what a world we live in..
                try:
                    for at in res.get_atoms():
                        at.set_bfactor(fitness_data[res_number])
                except KeyError:
                    # this is normal in most cases, a handful of residues will be missing from mutation data.
                    self.logger.warning(
                        f"No fitness score found for residue {res_number} of protein."
                    )

            # there's no biopython -> rdkit, so save to a PDB file and load it with RDKit.
            io = PDBIO()
            io.set_structure(protein_biopython)
            io.save(f"{str(protein)}_tmp")
            self.protein = Chem.MolFromPDBFile(f"{str(protein)}_tmp")
            os.remove(f"{str(protein)}_tmp")  # cleanup

        self.debug = debug
        if self.debug:
            self.logger.SetLevel(logging.DEBUG)
            self.logger.debug("Running in debug mode")
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
        protein_pdb = Chem.MolToPDBBlock(self.protein)
        # if there is an END line, remove it
        for line in protein_pdb.split("\n"):
            if line.startswith("END"):
                protein_pdb = protein_pdb.replace(line, "")
        mol_pdb = Chem.MolToPDBBlock(pose)
        joint_pdb = protein_pdb + mol_pdb
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
