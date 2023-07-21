import logging
from pathlib import Path
from typing import List, Optional, Union  # noqa: F401

from asapdiscovery.data.logging import FileLogger
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

        self.protein = Chem.MolFromPDBFile(str(protein))

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
        # colour uses target name
        protein_name = VizTargets.get_target_name(self.target, underscore=True)
        colour = getattr(HTMLBlockData, f"colour_{protein_name}")

        # orient uses full name with underscore
        target_ = VizTargets.get_name_underscore(self.target)
        orient_tail = getattr(HTMLBlockData, f"orient_tail_{target_}")

        return colour + orient_tail
