"""
The purpose of this module is to provide a set of standard functionality to analyze the docking results.
"""
from asapdiscovery.data.schema_v2 import complex, ligand, target
from asapdiscovery.docking.docking_data_validation import DockingResultCols
import pandas as pd


class DockingResults:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.complexes = self.get_complexes()
        self.ligands = self.get_ligands()
        self.targets = self.get_targets()
        self.docking_result_cols = DockingResultCols
        self.score_columns = self.get_score_columns()

    def get_complexes(self) -> list[complex]:
        """
        Returns the list of complexes in the docking results
        """
        return self.df[self.DockingResultCols.DU_STRUCTURE.value].unique()

    def get_ligands(self) -> list[ligand]:
        """
        Returns the list of ligands in the docking results
        """
        return self.df[self.DockingResultCols.LIGAND_ID.value].unique()

    def get_targets(self) -> list[target]:
        """
        Returns the list of targets in the docking results
        """
        return self.df[self.DockingResultCols.TARGET_ID.value].unique()

    def get_score_columns(self) -> list[str]:
        """
        Returns the list of score columns in the docking results
        """
        return [col for col in self.df.columns if col.startswith("docking-score-")]
