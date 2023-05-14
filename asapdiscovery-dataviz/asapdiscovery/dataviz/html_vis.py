from typing import List, Union, Optional
from rdkit import Chem
from pathlib import Path

from .html_blocks import (
    visualisation_header,
    colour_sars2,
    colour_mers,
    colour_7ene,
    orient_tail_sars2,
    orient_tail_mers,
    orient_tail_7ene,
    make_core_html,
)


def _load_first_molecule(file_path: Path):
    mols = Chem.SDMolSupplier(file_path)
    return mols[0]


class HTMLVisualiser:
    """
    Class for generating HTML visualisations of poses.
    """

    allowed_targets = ("sars2", "mers", "7ene")

    # TODO: replace input with a schema rather than paths.
    def __init__(
        self, poses: List[Path], paths: List[Path], target: str, protein: Path
    ):
        """
        Parameters
        ----------
        poses : List[Path]
            List of poses to visualise.
        paths : List[Path]
            List of paths to write the visualisations to.
        target : str
            Target to visualise poses for. Must be one of: "sars2", "mers", "7ene".

        """
        if not len(poses) == len(paths):
            raise ValueError("Number of poses and paths must be equal.")
        self.poses = [_load_first_molecule(pose) for pose in poses]
        if target not in self.allowed_targets:
            raise ValueError("Target must be one of: {}".format(self.allowed_targets))
        self.paths = paths
        self.target = target
        self.protein = Chem.MolFromPDBFile(str(protein))

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

    def write_pose_visualisations(self):
        """
        Write HTML visualisations for all poses.
        """
        for pose, path in zip(self.poses, self.paths):
            self.write_pose_visualisation(pose, path)

    def write_pose_visualisation(self, pose, path):
        """
        Write HTML visualisation for a single pose.
        """
        html = self.get_html(pose)
        self.write_html(html, path)

    def get_html(self, pose):
        """
        Get HTML for visualising a single pose.
        """
        return self.get_html_body(pose) + self.get_html_footer()

    def get_html_body(self, pose):
        """
        Get HTML body for pose visualisation
        """
        protein_pdb = Chem.MolToPDBBlock(self.protein)
        mol_pdb = Chem.MolToPDBBlock(pose)
        joint_pdb = protein_pdb + mol_pdb
        html_body = make_core_html(joint_pdb)
        return html_body

    def get_html_footer(self):
        """
        Get HTML footer for pose visualisation
        """
        if self.target == "sars2":
            return colour_sars2 + orient_tail_sars2
        elif self.target == "mers":
            return colour_mers + orient_tail_mers
        elif self.target == "7ene":
            return colour_7ene + orient_tail_7ene
