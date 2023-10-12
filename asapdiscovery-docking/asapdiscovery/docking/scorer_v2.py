import abc
from enum import Enum
from typing import Literal, Optional, Union, ClassVar

import dask
import pandas as pd
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.openeye import oedocking
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.docking.docking_v2 import DockingResult
from asapdiscovery.ml.inference import InferenceBase, get_inference_cls_from_model_type
from asapdiscovery.ml.models.ml_models import MLModelType
from pydantic import BaseModel, Field


class ScoreType(str, Enum):
    """
    Enum for score types.
    """

    chemgauss4 = "chemgauss4"
    schnet = "schnet"
    GAT = "GAT"


class Score(BaseModel):
    """
    Result of scoring.
    """

    score_type: ScoreType
    score: float


class ScorerBase(BaseModel):
    """
    Base class for docking.
    """

    @abc.abstractmethod
    def _score() -> list[DockingResult]:
        ...

    def score(
        self, inputs: list[DockingResult], use_dask: bool = False, dask_client=None
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
        return pd.DataFrame([score.dict() for score in scores])


class ChemGauss4Scorer(ScorerBase):
    """
    Scoring using ChemGauss.
    """

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            posed_mol = inp.posed_ligand.to_oemol()
            pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
            du = inp.input_pair.complex.target.to_oedu()
            pose_scorer.Initialize(du)
            chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
            results.append(
                Score(score_type=ScoreType.chemgauss4, score=chemgauss_score)
            )
        return results


class MLModelScorer(ScorerBase):
    """
    Score from some kind of ML model
    """

    model_type: ClassVar[MLModelType.GAT] = MLModelType.GAT
    targets: set[TargetTags] = Field(
        ..., description="Which targets can this model do predictions for"
    )
    model_name: Optional[str] = Field(
        None, description="String indicating which model to use"
    )
    inference_cls: Optional[InferenceBase] = Field(None)

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

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp.posed_ligand.smiles)
            results.append(Score(score_type=ScoreType.GAT, score=gat_score))
        return results


class SchnetScorer(MLModelScorer):
    """
    Scoring using Schnet ML Model
    """

    model_type: ClassVar[MLModelType.schnet] = MLModelType.schnet

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            schnet_score = self.inference_cls.predict_from_oemol(inp.to_posed_oemol())
            results.append(Score(score_type=ScoreType.SCHNET, score=schnet_score))
        return results


class MetaScorer(ScorerBase):
    """
    Score from a combination of other scorers
    """

    scorers: list[ScorerBase] = Field(..., description="Scorers to score with")

    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            scores = []
            for scorer in self.scorers:
                scores.extend(scorer.score(inputs=[inp]))
            results.append(scores)

        return results
