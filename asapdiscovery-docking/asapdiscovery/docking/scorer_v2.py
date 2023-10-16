import abc
from enum import Enum
from typing import ClassVar, Optional

import dask
import numpy as np
import pandas as pd
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.openeye import oedocking, save_openeye_pdb
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.schema_v2.ligand import LigandIdentifiers
from asapdiscovery.data.schema_v2.target import TargetIdentifiers
from asapdiscovery.docking.docking_v2 import DockingResult
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.ml.inference import InferenceBase, get_inference_cls_from_model_type
from asapdiscovery.ml.models.ml_models import MLModelType
from pydantic import BaseModel, Field


class ScoreType(str, Enum):
    """
    Enum for score types.
    """

    chemgauss4 = "chemgauss4"
    GAT = "GAT"
    schnet = "schnet"
    INVALID = "INVALID"


class ScoreUnits(str, Enum):
    """
    Enum for score units.
    """

    arbitrary = "arbitrary"
    kcal_mol = "kcal/mol"
    pIC50 = "pIC50"
    INVALID = "INVALID"


# this can possibly be done with subclasses and some aliases, but will do for now

_SCORE_MANIFOLD_ALIAS = {
    ScoreType.chemgauss4: DockingResultCols.DOCKING_SCORE_POSIT.value,
    ScoreType.GAT: DockingResultCols.COMPUTED_GAT_PIC50.value,
    ScoreType.schnet: DockingResultCols.COMPUTED_SCHNET_PIC50.value,
    ScoreType.INVALID: None,
    "target_name": DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
    "compound_name": DockingResultCols.LIGAND_ID.value,
}


class Score(BaseModel):
    """
    Result of scoring, we don't embed the input result because it can be large,
    instead we just store the input result ids.
    """

    score_type: ScoreType
    score: float
    compound_name: Optional[str]
    smiles: Optional[str]
    ligand_identifiers: Optional[LigandIdentifiers]
    target_name: Optional[str]
    target_identifiers: Optional[TargetIdentifiers]
    complex_ligand_smiles: Optional[str]
    probability: Optional[float]
    units: ScoreUnits

    @classmethod
    def from_score_and_docking_result(
        cls,
        score: float,
        score_type: ScoreType,
        units: ScoreUnits,
        docking_result: DockingResult,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=docking_result.posed_ligand.compound_name,
            smiles=docking_result.posed_ligand.smiles,
            ligand_ids=docking_result.posed_ligand.ids,
            target_name=docking_result.input_pair.complex.target.target_name,
            target_ids=docking_result.input_pair.complex.target.ids,
            complex_ligand_smiles=docking_result.input_pair.complex.ligand.smiles,
            probability=docking_result.probability,
            units=units,
        )

    @staticmethod
    def _combine_and_pivot_scores_df(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """ """
        df = pd.concat(dfs)
        indices = set(df.columns) - {"score_type", "score", "units"}
        df = df.pivot(
            index=indices,
            columns="score_type",
            values="score",
        ).reset_index()

        df.rename(columns=_SCORE_MANIFOLD_ALIAS, inplace=True)
        return df


class ScorerBase(BaseModel):
    """
    Base class for docking.
    """

    score_type: ClassVar[ScoreType.INVALID] = ScoreType.INVALID
    score_units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    @abc.abstractmethod
    def _score() -> list[DockingResult]:
        ...

    def score(
        self,
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        return_df: bool = False,
    ) -> list[Score]:
        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(self._score)(inputs=[inp])
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client=dask_client
            )
        else:
            outputs = self._score(inputs=inputs)

        if return_df:
            return self.scores_to_df(outputs)
        else:
            return outputs

    @staticmethod
    def scores_to_df(scores: list[Score]) -> pd.DataFrame:
        """
        Convert a list of scores to a dataframe.

        Parameters
        ----------
        scores : list[Score]
            List of scores

        Returns
        -------
        pd.DataFrame
            Dataframe of scores
        """
        # gather some fields from the input
        data_list = []
        # flatten the list of scores
        scores = np.ravel(scores)

        for score in scores:
            dct = score.dict()
            dct["score_type"] = score.score_type.value  # convert to string
            data_list.append(dct)
        # convert to a dataframe
        df = pd.DataFrame(data_list)

        return df


class ChemGauss4Scorer(ScorerBase):
    """
    Scoring using ChemGauss.
    """

    score_type: ClassVar[ScoreType.chemgauss4] = ScoreType.chemgauss4
    units: ClassVar[ScoreUnits.arbitrary] = ScoreUnits.arbitrary

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            posed_mol = inp.posed_ligand.to_oemol()
            pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
            du = inp.input_pair.complex.target.to_oedu()
            pose_scorer.Initialize(du)
            chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
            results.append(
                Score.from_score_and_docking_result(
                    chemgauss_score, self.score_type, self.units, inp
                )
            )
        return results


class MLModelScorer(ScorerBase):
    """
    Score from some kind of ML model
    """

    model_type: ClassVar[MLModelType.INVALID] = MLModelType.INVALID
    score_type: ClassVar[ScoreType.INVALID] = ScoreType.INVALID
    units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    targets: set[TargetTags] = Field(
        ..., description="Which targets can this model do predictions for"
    )
    model_name: str = Field(..., description="String indicating which model to use")
    inference_cls: InferenceBase = Field(..., description="Inference class")

    @classmethod
    def from_latest_by_target(cls, target: TargetTags):
        if cls.model_type == MLModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_latest_by_target(target)
        return cls(
            targets=inference_instance.targets,
            model_name=inference_instance.model_name,
            inference_cls=inference_instance,
        )

    @classmethod
    def from_model_name(cls, model_name: str):
        if cls.model_type == MLModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_model_name(model_name)
        return cls(
            targets=inference_instance.targets,
            model_name=inference_instance.model_name,
            inference_cls=inference_instance,
        )


class GATScorer(MLModelScorer):
    """
    Scoring using GAT ML Model
    """

    model_type: ClassVar[MLModelType.GAT] = MLModelType.GAT
    score_type: ClassVar[ScoreType.GAT] = ScoreType.GAT
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp.posed_ligand.smiles)
            results.append(
                Score.from_score_and_docking_result(
                    gat_score, self.score_type, self.units, inp
                )
            )
        return results


class SchnetScorer(MLModelScorer):
    """
    Scoring using Schnet ML Model
    """

    model_type: ClassVar[MLModelType.schnet] = MLModelType.schnet
    score_type: ClassVar[ScoreType.schnet] = ScoreType.schnet
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            schnet_score = self.inference_cls.predict_from_oemol(inp.to_posed_oemol())
            # save_openeye_pdb(inp.to_posed_oemol(), inp.input_pair.complex.target.target_name + "_+_" + inp.posed_ligand.compound_name + ".pdb")
            results.append(
                Score.from_score_and_docking_result(
                    schnet_score, self.score_type, self.units, inp
                )
            )
        return results


class MetaScorer(BaseModel):
    """
    Score from a combination of other scorers
    """

    scorers: list[ScorerBase] = Field(..., description="Scorers to score with")

    def score(
        self,
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        return_df: bool = False,
    ) -> list[Score]:
        results = []
        for scorer in self.scorers:
            vals = scorer.score(
                inputs=inputs,
                use_dask=use_dask,
                dask_client=dask_client,
                return_df=return_df,
            )
            results.append(vals)

        if return_df:
            return Score._combine_and_pivot_scores_df(results)

        return np.ravel(results).tolist()
