import abc
import logging
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional, Union

import numpy as np
import pandas as pd
from asapdiscovery.data.backend.openeye import oedocking
from asapdiscovery.data.backend.plip import compute_fint_score
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand, LigandIdentifiers
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
from asapdiscovery.ml.models import ASAPMLModelRegistry
from mtenn.config import ModelType
from multimethod import multimethod
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

    @classmethod
    def from_score_and_complex(
        cls,
        score: float,
        score_type: ScoreType,
        units: ScoreUnits,
        complex: Complex,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=complex.ligand.compound_name,
            smiles=complex.ligand.smiles,
            ligand_inchikey=complex.ligand.inchikey,
            ligand_ids=complex.ligand.ids,
            target_name=complex.target.target_name,
            target_ids=complex.target.ids,
            complex_ligand_smiles=complex.ligand.smiles,
            probability=None,
            units=units,
        )

    @classmethod
    def from_score_and_smiles(
        cls,
        score: float,
        smiles: str,
        score_type: ScoreType,
        units: ScoreUnits,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=None,
            smiles=smiles,
            ligand_inchikey=None,
            ligand_ids=None,
            target_name=None,
            target_ids=None,
            complex_ligand_smiles=None,
            probability=None,
            units=units,
        )

    @classmethod
    def from_score_and_ligand(
        cls,
        score: float,
        ligand: Ligand,
        score_type: ScoreType,
        units: ScoreUnits,
    ):
        return cls(
            score_type=score_type,
            score=score,
            compound_name=ligand.compound_name,
            smiles=ligand.smiles,
            ligand_inchikey=ligand.inchikey,
            ligand_ids=ligand.ids,
            target_name=None,
            target_ids=None,
            complex_ligand_smiles=None,
            probability=None,
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
    Base class for scoring functions.
    """

    score_type: ClassVar[ScoreType.INVALID] = ScoreType.INVALID
    score_units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    @abc.abstractmethod
    def _score() -> list[DockingResult]: ...

    def score(
        self,
        inputs: Union[
            list[DockingResult], list[Complex], list[Path], list[str], list[Ligand]
        ],
        use_dask: bool = False,
        dask_client=None,
        dask_failure_mode=DaskFailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        return_df: bool = False,
    ) -> list[Score]:
        """
        Score the inputs. Most of the work is done in the _score method, this method is in turn dispatched based on type to various methods.


        Parameters
        ----------
        inputs : Union[list[DockingResult], list[Complex], list[Path], list[str], list[Ligand]],
            List of inputs to score
        use_dask : bool, optional
            Whether to use dask, by default False
        dask_client : dask.distributed.Client, optional
            Dask client to use, by default None
        dask_failure_mode : DaskFailureMode, optional
            How to handle dask failures, by default DaskFailureMode.SKIP
        backend : BackendType, optional
            Backend to use, by default BackendType.IN_MEMORY
        reconstruct_cls : Optional[Callable], optional
            Function to use to reconstruct the inputs, by default None
        return_df : bool, optional
            Whether to return a dataframe, by default False

        """
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

    def __hash__(self) -> int:
        return self.score_type.__hash__()

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

    Overloaded to accept DockingResults, Complexes, or Paths to PDB files.

    """

    score_type: ClassVar[ScoreType.chemgauss4] = ScoreType.chemgauss4
    units: ClassVar[ScoreUnits.arbitrary] = ScoreUnits.arbitrary

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(self, inputs) -> list[Score]:
        """
        Score the inputs, dispatching based on type.
        """
        return self._dispatch(inputs)

    @multimethod
    def _dispatch(self, inputs: list[DockingResult], **kwargs) -> list[Score]:
        """
        Dispatch for DockingResults
        """
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

    @_dispatch.register
    def _dispatch(self, inputs: list[Complex], **kwargs) -> list[Score]:
        """
        Dispatch for Complexes
        """
        results = []
        for inp in inputs:
            posed_mol = inp.ligand.to_oemol()
            pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
            receptor = inp.target.to_oemol()
            pose_scorer.Initialize(receptor)
            chemgauss_score = pose_scorer.ScoreLigand(posed_mol)
            results.append(
                Score.from_score_and_complex(
                    chemgauss_score, self.score_type, self.units, inp
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Path], **kwargs) -> list[Score]:
        """
        Dispatch for PDB files from disk
        """
        # assuming reading PDB files from disk
        complexes = [
            Complex.from_pdb(
                p,
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
                target_kwargs={"target_name": f"{p.stem}_target"},
            )
            for p in inputs
        ]
        return self._dispatch(complexes)


class FINTScorer(ScorerBase):
    """
    Score using Fitness Interaction Score

    Overloaded to accept DockingResults, Complexes, or Paths to PDB files.
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
    def _score(
        self, inputs: Union[list[DockingResult], list[Complex], list[Path]]
    ) -> list[Score]:
        """
        Score the inputs, dispatching based on type.
        """
        return self._dispatch(inputs)

    @multimethod
    def _dispatch(self, inputs: list[DockingResult], **kwargs) -> list[Score]:
        """
        Dispatch for DockingResults
        """
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

    @_dispatch.register
    def _dispatch(self, inputs: list[Complex], **kwargs):
        """
        Dispatch for Complexes
        """
        results = []
        for inp in inputs:
            _, fint_score = compute_fint_score(
                inp.target.to_oemol(), inp.ligand.to_oemol(), self.target
            )
            results.append(
                Score.from_score_and_complex(
                    fint_score, self.score_type, self.units, inp
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Path], **kwargs):
        """
        Dispatch for PDB files from disk
        """
        # assuming reading PDB files from disk
        complexes = [
            Complex.from_pdb(
                p,
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
                target_kwargs={"target_name": f"{p.stem}_target"},
            )
            for p in inputs
        ]

        return self._dispatch(complexes)


# keep track of all the ml scorers
_ml_scorer_classes_meta = []


# decorator to register all the ml scorers
def register_ml_scorer(cls):
    _ml_scorer_classes_meta.append(cls)
    return cls


class MLModelScorer(ScorerBase):
    """
    Baseclass to score from some kind of ML model, including 2D or 3D models
    """

    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    score_type: ClassVar[ScoreType.INVALID] = ScoreType.INVALID
    units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    targets: set[TargetTags] = Field(
        ..., description="Which targets can this model do predictions for"
    )
    model_name: str = Field(..., description="String indicating which model to use")
    inference_cls: InferenceBase = Field(..., description="Inference class")

    # make sure we can hash this
    def __hash__(self) -> int:
        return self.model_name.__hash__()

    @classmethod
    def from_latest_by_target(
        cls, target: TargetTags, error: str = "skip"
    ) -> Optional["MLModelScorer"]:
        """
        Get the latest model for a target and a model type and return the scorer

        Parameters
        ----------
        target : TargetTags
            Target to get model for
        error : str, optional
            What to do if no model is found, by default "skip" and return None

        Returns
        -------
        MLModelScorer
            Scorer for the model
        """

        if cls.model_type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        inference_cls = get_inference_cls_from_model_type(cls.model_type)
        inference_instance = inference_cls.from_latest_by_target(target)
        if inference_instance is None:
            estring = f"no ML model of type {cls.model_type} found for target: {target}"
            if error == "raise":
                raise Exception(estring)
            elif error == "skip":
                logger.warn(estring)
                return None
            else:
                raise ValueError(f"error must be 'raise' or 'skip', got {error}")
        else:
            return cls(
                targets=inference_instance.targets,
                model_name=inference_instance.model_name,
                inference_cls=inference_instance,
            )

    @classmethod
    def from_model_name(
        cls, model_name: str, error: str = "skip"
    ) -> Optional["MLModelScorer"]:
        """
        Get a scorer from a specific model name

        Parameters
        ----------
        model_name : str
            Name of the model to get
        error : str, optional
            What to do if no model is found, by default "skip" and return None

        Returns
        -------
        MLModelScorer
        """
        if cls.model_type == ModelType.INVALID:
            model_type = ASAPMLModelRegistry.get_model_type_from_name(model_name)
        else:
            model_type = cls.model_type
        inference_cls = get_inference_cls_from_model_type(model_type)
        inference_instance = inference_cls.from_model_name(model_name)
        scorer_cls = get_ml_scorer_cls_from_model_type(model_type)
        if inference_instance is None:
            estring = (
                f"no ML model of name {model_name} found for model type: {model_type}"
            )
            if error == "raise":
                raise Exception(estring)
            elif error == "skip":
                logger.warn(estring)
                return None
            else:
                raise ValueError(f"error must be 'raise' or 'skip', got {error}")
        return scorer_cls(
            targets=inference_instance.targets,
            model_name=inference_instance.model_name,
            inference_cls=inference_instance,
        )

    @staticmethod
    def autoselect_by_target(target: TargetTags) -> list["MLModelScorer"]:
        """
        Get the latest models of EACH TYPE for a target and return the scorers

        Parameters
        ----------
        target : TargetTags
            Target to get models for

        Returns
        -------
        list[MLModelScorer]
            List of scorers, one for each model type available for the target
        """
        models = ASAPMLModelRegistry.get_latest_models_for_target(target)
        scorer_classes = []
        for model in models:
            ml_scorer_class = get_ml_scorer_cls_from_model_type(model.type)
            scorer_classes.append(ml_scorer_class.from_model_name(model.name))
        return scorer_classes

    @staticmethod
    def from_latest_by_target_and_type(target: TargetTags, type: ModelType):
        """
        Get the latest ML Scorer by target and type.

        Parameters
        ----------
        target : TargetTags
            Target to get the scorer for
        type : ModelType
            Type of model to get the scorer for
        """
        if type == ModelType.INVALID:
            raise Exception("trying to instantiate some kind a baseclass")
        scorer_class = get_ml_scorer_cls_from_model_type(type)
        return scorer_class.from_latest_by_target(target)


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
    def _score(
        self, inputs: Union[list[DockingResult], list[str], list[Ligand]]
    ) -> list[Score]:
        """
        Score the inputs, dispatching based on type.
        """
        return self._dispatch(inputs)

    @multimethod
    def _dispatch(self, inputs: list[DockingResult], **kwargs) -> list[Score]:
        """
        Dispatch for DockingResults
        """
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp.posed_ligand.smiles)
            results.append(
                Score.from_score_and_docking_result(
                    gat_score, self.score_type, self.units, inp
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[str], **kwargs) -> list[Score]:
        """
        Dispatch for SMILES strings
        """
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp)
            results.append(
                Score.from_score_and_smiles(
                    gat_score,
                    inp,
                    self.score_type,
                    self.units,
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Ligand], **kwargs) -> list[Score]:
        """
        Dispatch for Ligands
        """
        results = []
        for inp in inputs:
            gat_score = self.inference_cls.predict_from_smiles(inp.smiles)
            results.append(
                Score.from_score_and_ligand(
                    gat_score,
                    inp,
                    self.score_type,
                    self.units,
                )
            )
        return results


class E3MLModelScorer(MLModelScorer):
    """
    Scoring using ML Models that operate over 3D structures
    These all share an interface so we can use multimethods to dispatch
    for the different input types for all subclasses.
    """

    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    score_type: ClassVar[ScoreType.INVALID] = ScoreType.INVALID
    units: ClassVar[ScoreUnits.INVALID] = ScoreUnits.INVALID

    @dask_vmap(["inputs"])
    @backend_wrapper("inputs")
    def _score(
        self, inputs: Union[list[DockingResult], list[Complex], list[Path]]
    ) -> list[Score]:
        return self._dispatch(inputs)

    @multimethod
    def _dispatch(self, inputs: list[DockingResult], **kwargs) -> list[Score]:
        results = []
        for inp in inputs:
            score = self.inference_cls.predict_from_oemol(inp.to_posed_oemol())
            results.append(
                Score.from_score_and_docking_result(
                    score, self.score_type, self.units, inp
                )
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Complex], **kwargs) -> list[Score]:
        results = []
        for inp in inputs:
            score = self.inference_cls.predict_from_oemol(inp.to_combined_oemol())
            results.append(
                Score.from_score_and_complex(score, self.score_type, self.units, inp)
            )
        return results

    @_dispatch.register
    def _dispatch(self, inputs: list[Path], **kwargs) -> list[Score]:
        # assuming reading PDB files from disk
        complexes = [
            Complex.from_pdb(
                p,
                ligand_kwargs={"compound_name": f"{p.stem}_ligand"},
                target_kwargs={"target_name": f"{p.stem}_target"},
            )
            for i, p in enumerate(inputs)
        ]
        return self._dispatch(complexes)


@register_ml_scorer
class SchnetScorer(E3MLModelScorer):
    """
    Scoring using Schnet ML Model
    """

    model_type: ClassVar[ModelType.schnet] = ModelType.schnet
    score_type: ClassVar[ScoreType.schnet] = ScoreType.schnet
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50


@register_ml_scorer
class E3NNScorer(E3MLModelScorer):
    """
    Scoring using e3nn ML Model
    """

    model_type: ClassVar[ModelType.e3nn] = ModelType.e3nn
    score_type: ClassVar[ScoreType.e3nn] = ScoreType.e3nn
    units: ClassVar[ScoreUnits.pIC50] = ScoreUnits.pIC50


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
    Score from a combination of other scorers, the scorers must share an input type,
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
        """
        Score the inputs using all the scorers provided in the constructor
        """
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
