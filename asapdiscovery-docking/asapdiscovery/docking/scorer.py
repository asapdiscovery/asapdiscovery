import abc
import logging
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union

import numpy as np
import pandas as pd
from asapdiscovery.data.backend.openeye import oedocking
from asapdiscovery.data.backend.plip import compute_fint_score
from asapdiscovery.data.schema.ligand import LigandIdentifiers
from asapdiscovery.data.schema.target import TargetIdentifiers
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.data.util.dask_utils import (
    BackendType,
    DaskFailureMode,
    backend_wrapper,
    dask_vmap,
)
from asapdiscovery.docking.docking import DockingResult
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.genetics.fitness import target_has_fitness_data
from asapdiscovery.ml.inference import InferenceBase, get_inference_cls_from_model_type
from mtenn.config import ModelType
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ScoreType(str, Enum):
    """
    Enum for score types.
    """

    chemgauss4 = "chemgauss4"
    FINT = "FINT"
    GAT = "GAT"
    schnet = "schnet"
    e3nn = "e3nn"
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
    ScoreType.FINT: DockingResultCols.FITNESS_SCORE_FINT.value,
    ScoreType.GAT: DockingResultCols.COMPUTED_GAT_PIC50.value,
    ScoreType.schnet: DockingResultCols.COMPUTED_SCHNET_PIC50.value,
    ScoreType.e3nn: DockingResultCols.COMPUTED_E3NN_PIC50.value,
    ScoreType.INVALID: None,
    "target_name": DockingResultCols.DOCKING_STRUCTURE_POSIT.value,
    "compound_name": DockingResultCols.LIGAND_ID.value,
    "smiles": DockingResultCols.SMILES.value,
    "ligand_inchikey": DockingResultCols.INCHIKEY.value,
    "probability": DockingResultCols.DOCKING_CONFIDENCE_POSIT.value,
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
    ligand_inchikey: Optional[str]
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
            ligand_inchikey=docking_result.posed_ligand.inchikey,
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
    def _score() -> list[DockingResult]: ...

    def score(
        self,
        inputs: Union[list[DockingResult], list[Path]],
        use_dask: bool = False,
        dask_client=None,
        dask_failure_mode=DaskFailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        return_df: bool = False,
    ) -> list[Score]:
        outputs = self._score(
            inputs=inputs,
            use_dask=use_dask,
            dask_client=dask_client,
            dask_failure_mode=dask_failure_mode,
            backend=backend,
            reconstruct_cls=reconstruct_cls,
        )

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

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
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


class FINTScorer(ScorerBase):
    """
    Score using Fitness Interaction Score
    """

    score_type: ClassVar[ScoreType.FINT] = ScoreType.FINT
    units: ClassVar[ScoreUnits.arbitrary] = ScoreUnits.arbitrary
    target: TargetTags = Field(..., description="Which target to use for scoring")

    @validator("target")
    @classmethod
    def validate_target(cls, v):
        if not target_has_fitness_data(v):
            raise ValueError(
                "target does not have fitness data so cannot use FINTScorer"
            )
        return v

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            _, fint_score = compute_fint_score(
                inp.to_protein(), inp.posed_ligand.to_oemol(), self.target
            )
            results.append(
                Score.from_score_and_docking_result(
                    fint_score, self.score_type, self.units, inp
                )
            )
        return results


_ml_scorer_classes_meta = []


# decorator to register all the ml scorers
def register_ml_scorer(cls):
    _ml_scorer_classes_meta.append(cls)
    return cls


class MLModelScorer(ScorerBase):
    """
    Score from some kind of ML model
    """

    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    score_type: ClassVar[ScoreType.INVALID] = ScoreType.INVALID
    units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    targets: set[TargetTags] = Field(
        ..., description="Which targets can this model do predictions for"
    )
    model_name: str = Field(..., description="String indicating which model to use")
    inference_cls: InferenceBase = Field(..., description="Inference class")

    @classmethod
    def from_latest_by_target(cls, target: TargetTags):
        if cls.model_type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_latest_by_target(target)
        if inference_instance is None:
            logger.warn(
                f"no ML model of type {cls.model_type} found for target: {target}, skipping"
            )
            return None
        else:
            return cls(
                targets=inference_instance.targets,
                model_name=inference_instance.model_name,
                inference_cls=inference_instance,
            )

    @staticmethod
    def from_latest_by_target_and_type(target: TargetTags, type: ModelType):
        if type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        scorer_class = get_ml_scorer_cls_from_model_type(type)
        return scorer_class.from_latest_by_target(target)

    @classmethod
    def from_model_name(cls, model_name: str):
        if cls.model_type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_model_name(model_name)
        return cls(
            targets=inference_instance.targets,
            model_name=inference_instance.model_name,
            inference_cls=inference_instance,
        )


@register_ml_scorer
class GATScorer(MLModelScorer):
    """
    Scoring using GAT ML Model
    """

    model_type: ClassVar[ModelType.GAT] = ModelType.GAT
    score_type: ClassVar[ScoreType.GAT] = ScoreType.GAT
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
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


@register_ml_scorer
class SchnetScorer(MLModelScorer):
    """
    Scoring using Schnet ML Model
    """

    model_type: ClassVar[ModelType.schnet] = ModelType.schnet
    score_type: ClassVar[ScoreType.schnet] = ScoreType.schnet
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            schnet_score = self.inference_cls.predict_from_oemol(inp.to_posed_oemol())
            results.append(
                Score.from_score_and_docking_result(
                    schnet_score, self.score_type, self.units, inp
                )
            )
        return results


@register_ml_scorer
class E3NNScorer(MLModelScorer):
    """
    Scoring using e3nn ML Model
    """

    model_type: ClassVar[ModelType.e3nn] = ModelType.e3nn
    score_type: ClassVar[ScoreType.e3nn] = ScoreType.e3nn
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(self, inputs: list[DockingResult]) -> list[Score]:
        results = []
        for inp in inputs:
            e3nn_score = self.inference_cls.predict_from_oemol(inp.to_posed_oemol())
            results.append(
                Score.from_score_and_docking_result(
                    e3nn_score, self.score_type, self.units, inp
                )
            )
        return results


def get_ml_scorer_cls_from_model_type(model_type: ModelType):
    instantiable_classes = [
        m for m in _ml_scorer_classes_meta if m.model_type != ModelType.INVALID
    ]
    scorer_class = [m for m in instantiable_classes if m.model_type == model_type]
    if len(scorer_class) != 1:
        raise Exception("Somehow got multiple scorers")
    return scorer_class[0]


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
        dask_failure_mode=DaskFailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        return_df: bool = False,
    ) -> list[Score]:
        results = []
        for scorer in self.scorers:
            vals = scorer.score(
                inputs=inputs,
                use_dask=use_dask,
                dask_client=dask_client,
                dask_failure_mode=dask_failure_mode,
                backend=backend,
                reconstruct_cls=reconstruct_cls,
                return_df=return_df,
            )
            results.append(vals)

        if return_df:
            return Score._combine_and_pivot_scores_df(results)

        return np.ravel(results).tolist()
