import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union  # noqa: F401

import dgl
import mtenn
import numpy as np
import torch
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.ml.config import DatasetConfig
from asapdiscovery.ml.dataset import DockedDataset, GraphDataset
from asapdiscovery.ml.models import (
    ASAPMLModelRegistry,
    LocalMLModelSpecBase,
    MLModelRegistry,
    MLModelSpec,
    MLModelSpecBase,
)

# static import of models from base yaml here
from dgllife.utils import CanonicalAtomFeaturizer
from mtenn.config import E3NNModelConfig, GATModelConfig, ModelType, SchNetModelConfig
from pydantic.v1 import BaseModel, Field

"""
TODO

Need to adjust inference model construction to use new ModelConfigs. Can create one for
each model and store in S3 to use during testing.
"""


class InferenceBase(BaseModel):
    class Config:
        validate_assignment = True
        allow_mutation = False
        arbitrary_types_allowed = True
        allow_extra = False

    targets: Optional[Any] = Field(
        None,
        description="Targets that them model can predict for",  # FIXME: should be Optional[Set[TargetTags]] but this causes issues with pydantic
    )
    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID
    model_name: str = Field(..., description="Name of model to use")
    model_spec: Optional[MLModelSpecBase] = Field(
        ..., description="Model spec used to create Model to use"
    )
    local_model_spec: LocalMLModelSpecBase = Field(
        ..., description="Local model spec used to create Model to use"
    )
    device: str = Field("cpu", description="Device to use for inference")
    models: Optional[list[torch.nn.Module]] = Field(..., description="PyTorch model(s)")

    @property
    def is_ensemble(self):
        return len(self.models) > 1

    @property
    def ensemble_size(self):
        return len(self.models)

    @classmethod
    def from_latest_by_target(
        cls,
        target: TargetTags,
        model_registry: MLModelRegistry = ASAPMLModelRegistry,
        **kwargs,
    ):
        """
        Create an InferenceBase object from the latest model for the latest target.

        Returns
        -------
        InferenceBase
            InferenceBase object created from latest model for latest target.
        """
        model_spec = model_registry.get_latest_model_for_target_and_type(
            target, cls.model_type
        )

        if model_spec is None:  # No model found, return None
            return None
        else:
            return cls.from_ml_model_spec(model_spec, **kwargs)

    @classmethod
    def from_latest_by_target_and_endpoint(
        cls,
        target: TargetTags,
        endpoint: str,
        model_registry: MLModelRegistry = ASAPMLModelRegistry,
        **kwargs,
    ):
        """
        Create an InferenceBase object from the latest model for the latest target.

        Returns
        -------
        InferenceBase
            InferenceBase object created from latest model for latest target.
        """
        model_spec = model_registry.get_latest_model_for_target_and_endpoint_and_type(
            target, endpoint, cls.model_type
        )

        if model_spec is None:
            return None
        else:
            return cls.from_ml_model_spec(model_spec, **kwargs)

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        model_registry: MLModelRegistry = ASAPMLModelRegistry,
        **kwargs,
    ):
        """
        Create an InferenceBase object from a model name.

        Returns
        -------
        InferenceBase
            InferenceBase object created from model name.
        """
        model_spec = model_registry.get_model(model_name)
        return cls.from_ml_model_spec(model_spec, **kwargs)

    @classmethod
    def from_ml_model_spec(
        cls,
        model_spec: MLModelSpec,
        device: str = "cpu",
        local_dir: Optional[Union[str, Path]] = None,
        build_model_kwargs: Optional[dict] = {},
    ) -> "InferenceBase":
        """
        Create an InferenceBase object from an MLModelSpec.

        Parameters
        ----------
        model_spec : MLModelSpec
            MLModelSpec to use to create InferenceBase object.

        Returns
        -------
        InferenceBase
            InferenceBase object created from MLModelSpec.
        """
        model_components = model_spec.pull(local_dir=local_dir)
        return cls.from_local_model_spec(
            model_components,
            device=device,
            model_spec=model_spec,
            build_model_kwargs=build_model_kwargs,
        )

    @classmethod
    def from_local_model_spec(
        cls,
        local_model_spec: LocalMLModelSpecBase,
        device: str = "cpu",
        model_spec: Optional[MLModelSpec] = None,
        build_model_kwargs: Optional[dict] = {},
    ) -> "InferenceBase":
        """
        Create an InferenceBase object from a LocalMLModelSpec.

        Parameters
        ----------
        local_model_spec : LocalMLModelSpec
            LocalMLModelSpec to use to create InferenceBase object.

        Returns
        -------
        InferenceBase
            InferenceBase object created from LocalMLModelSpec.
        """

        # First make sure mtenn versions are compatible
        if not local_model_spec.check_mtenn_version():
            lower_pin = (
                f">={local_model_spec.mtenn_lower_pin}"
                if local_model_spec.mtenn_lower_pin
                else ""
            )
            upper_pin = (
                f"<{local_model_spec.mtenn_upper_pin}"
                if local_model_spec.mtenn_upper_pin
                else ""
            )
            sep = "," if lower_pin and upper_pin else ""

            raise ValueError(
                f"Installed mtenn version ({mtenn.__version__}) "
                "is incompatible with the version specified in the MLModelSpec "
                f"({lower_pin}{sep}{upper_pin})"
            )

        # Select appropriate Config class
        match local_model_spec.type:
            case ModelType.GAT:
                config_cls = GATModelConfig
            case ModelType.schnet:
                config_cls = SchNetModelConfig
            case ModelType.e3nn:
                config_cls = E3NNModelConfig
            case other:
                raise ValueError(f"Can't instantiate model config for type {other}.")

        models = []

        if model_spec.ensemble:
            for model in local_model_spec.models:

                config_kwargs = json.loads(model.config_file.read_text())

                # warnings.warn(f"failed to parse model config file, {model.config_file}")
                # config_kwargs = {}
                config_kwargs["model_weights"] = torch.load(
                    model.weights_file, map_location=device
                )
                model = config_cls(**config_kwargs).build()
                model.eval()
                models.append(model)
        else:
            config_kwargs = json.loads(local_model_spec.config_file.read_text())
            config_kwargs["model_weights"] = torch.load(
                local_model_spec.weights_file, map_location=device
            )
            model = config_cls(**config_kwargs).build()
            model.eval()
            models.append(model)

        return cls(
            targets=local_model_spec.targets,
            model_type=local_model_spec.type,
            model_name=local_model_spec.name,
            model_spec=model_spec,
            local_model_spec=local_model_spec,
            device=device,
            models=models,
        )

    def predict(self, input_data, aggfunc=np.mean, errfunc=np.std, return_err=False):
        """Predict on data, needs to be overloaded in child classes most of
        the time

        Parameters
        ----------

        input_data: pytorch.Tensor
            Input data to predict on.
        aggfunc: function, default=np.mean
            Function to aggregate predictions from multiple models.
        errfunc: function, default=np.std
            Function to calculate error from multiple models.
        return_err: bool, default=False
            Return error in addition to prediction.

        Returns
        -------
        np.ndarray
            Prediction from model.
        float
            Error from model.
        """
        with torch.no_grad():

            # feed in data in whatever format is required by the model
            # for model ensemble, we need to loop through each model and get the
            # prediction from each, then aggregate
            input_tensor = torch.tensor(input_data).to(self.device)

            aggregate_preds = []
            for model in self.models:
                output_tensor = model(input_tensor)[0].cpu().numpy().flatten()
                aggregate_preds.append(output_tensor)
            if self.is_ensemble:
                aggregate_preds = np.array(aggregate_preds)
                pred = aggfunc(aggregate_preds, axis=0)
                err = errfunc(aggregate_preds, axis=0)
            else:
                # iterates only once, just return the prediction
                pred = output_tensor
                err = np.asarray([np.nan])
            if return_err:
                return pred, err
            else:
                return pred


class GATInference(InferenceBase):
    model_type: ClassVar[ModelType.GAT] = ModelType.GAT

    def predict(
        self, g: dgl.DGLGraph, aggfunc=np.mean, errfunc=np.std, return_err=False
    ):
        """Predict on a graph, requires a DGLGraph object with the `ndata`
        attribute `h` containing the node features. This is done by constucting
        the `GraphDataset` with the node_featurizer=`dgllife.utils.CanonicalAtomFeaturizer()`
        argument.


        Parameters
        ----------
        g : dgl.DGLGraph
            DGLGraph object.
        aggfunc: function, default=np.mean
            Function to aggregate predictions from multiple models.
        errfunc: function, default=np.std
            Function to calculate error from multiple models.
        return_err: bool, default=False
            Return error in addition to prediction.

        Returns
        -------
        np.ndarray
            Predictions for each graph.
        np.ndarray
            Errors for each prediction.
        """
        with torch.no_grad():
            aggregate_preds = []
            for model in self.models:
                output_tensor = model({"g": g})[0].cpu().numpy().flatten()
                # we ravel to always get a 1D array
                aggregate_preds.append(output_tensor)
            if self.is_ensemble:
                aggregate_preds = np.array(aggregate_preds).flatten()
                pred = aggfunc(aggregate_preds)
                err = errfunc(aggregate_preds)
            else:
                pred = output_tensor
                err = np.asarray([np.nan])

            if return_err:
                return pred, err
            else:
                return pred

    def predict_from_smiles(
        self,
        smiles: Union[str, list[str]],
        node_featurizer=None,
        edge_featurizer=None,
        return_err=False,
    ) -> Union[np.ndarray, float]:
        """Predict on a list of SMILES strings, or a single SMILES string.

        Parameters
        ----------
        smiles : Union[str, List[str]]
            SMILES string or list of SMILES strings.
        node_featurizer : BaseAtomFeaturizer, optional
            Featurizer for node data
        edge_featurizer : BaseBondFeaturizer, optional
            Featurizer for edges
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Predictions for each graph, or a single prediction if only one SMILES string is provided.
        np.ndarray or float
            Errors for each prediction, or a single error if only one SMILES string is provided.
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        ligands = [
            Ligand.from_smiles(smi, compound_name=f"eval_{i}")
            for i, smi in enumerate(smiles)
        ]

        if not node_featurizer:
            node_featurizer = CanonicalAtomFeaturizer()
        ds = GraphDataset.from_ligands(
            ligands, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer
        )
        # always return a 2D array, then we can mask out the err dimension
        data = [self.predict(pose["g"], return_err=True) for _, pose in ds]
        data = np.asarray(data, dtype=np.float32)
        # if it is 1D array, we need to convert to 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        preds = data[:, 0]
        if return_err:
            errs = data[:, 1]
        # return a scalar float value if we only have one input
        if np.all(np.array(preds.shape) == 1):
            preds = preds.item()
            if return_err:
                errs = errs.item()

        else:
            # flatten the array if we have multiple inputs
            preds = preds.flatten()
            if return_err:
                errs = errs.flatten()

        if return_err:
            return preds, errs
        else:
            return preds


class StructuralInference(InferenceBase):
    """
    Inference class for models that take a structure as input.
    """

    model_type: ClassVar[ModelType.INVALID] = ModelType.INVALID

    def predict(
        self, pose_dict: dict, aggfunc=np.mean, errfunc=np.std, return_err=False
    ):
        """Predict on a pose, requires a dictionary with the pose data with
        the keys: "z", "pos", "lig" with the required tensors in each

        Parameters
        ----------
        pose_dict : dict
            Dictionary with pose data.
        aggfunc: function, default=np.mean
            Function to aggregate predictions from multiple models.
        errfunc: function, default=np.std
            Function to calculate error from multiple models.
        return_err: bool, default=False


        Returns
        -------
        np.ndarray or float
            Predictions for the pose.
        np.ndarray or float
            Errors for the pose

        """
        with torch.no_grad():
            aggregate_preds = []
            for model in self.models:
                output_tensor = model(pose_dict)[0].cpu().numpy().flatten()
                # we ravel to always get a 1D array
                aggregate_preds.append(output_tensor)
            if self.is_ensemble:
                aggregate_preds = np.array(aggregate_preds).flatten()
                pred = aggfunc(aggregate_preds)
                err = errfunc(aggregate_preds)
            else:
                pred = output_tensor
                err = np.asarray([np.nan])

            if return_err:
                return pred, err
            else:
                return pred

    def predict_from_structure_file(
        self, pose: Union[Path, list[Path]], for_e3nn: bool = False, return_err=False
    ) -> Union[np.ndarray, float]:
        """Predict on a list of poses or a single pose.

        Parameters
        ----------
        pose : Union[Path, List[Path]]
            Path to pose file or list of paths to pose files.
        for_e3nn : bool, default=False
            If this prediction is being made for an e3nn model. Need to adjust the
            dict labels in this case
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Prediction for poses, or a single prediction if only one pose is provided.
        np.ndarray or float
            Error for poses, or a single error if only one pose is provided.
        """

        if isinstance(pose, Path):
            pose = [pose]

        complexes = [
            Complex.from_pdb(
                pdb_file=p,
                target_kwargs={"target_name": "pose"},
                ligand_kwargs={"compound_name": str(i)},
            )
            for i, p in enumerate(pose)
        ]
        pose = [DockedDataset._complex_to_pose(c) for c in complexes]
        if for_e3nn:
            pose = [
                p[1] for p in DatasetConfig.fix_e3nn_labels([(None, p) for p in pose])
            ]
        # always return a 2D array, then we can mask out the err dimension
        data = [self.predict(p, return_err=True) for p in pose]
        data = np.asarray(data, dtype=np.float32)
        # if it is 1D array, we need to convert to 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        preds = data[:, 0]
        if return_err:
            errs = data[:, 1]
        # return a scalar float value if we only have one input
        if np.all(np.array(preds.shape) == 1):
            preds = preds.item()
            if return_err:
                errs = errs.item()

        else:
            # flatten the array if we have multiple inputs
            preds = preds.flatten()
            if return_err:
                errs = errs.flatten()

        if return_err:
            return preds, errs
        else:
            return preds

    def predict_from_oemol(
        self,
        pose: Union[oechem.OEMol, list[oechem.OEMol]],
        for_e3nn: bool = False,
        return_err=False,
    ) -> Union[np.ndarray, float]:
        """
        Predict on a (list of) OEMol objects.

        Parameters
        ----------
        pose : Union[oechem.OEMol, list[oechem.OEMol]]
            (List of) OEMol pose(s)
        for_e3nn : bool, default=False
            If this prediction is being made for an e3nn model. Need to adjust the
            dict labels in this case
        return_err: bool, default=False

        Returns
        -------
        np.ndarray or float
            Model prediction(s)
        np.ndarray or float
            Model error(s)
        """
        if isinstance(pose, oechem.OEMolBase):
            pose = [pose]

        # Build each complex
        complexes = [
            Complex.from_oemol(
                complex_mol=p,
                target_kwargs={"target_name": "pose"},
                ligand_kwargs={"compound_name": str(i)},
            )
            for i, p in enumerate(pose)
        ]

        # Build each pose from complex
        pose = [DockedDataset._complex_to_pose(c) for c in complexes]
        if for_e3nn:
            pose = [
                p[1] for p in DatasetConfig.fix_e3nn_labels([(None, p) for p in pose])
            ]

        # Make predictions
        # always return a 2D array, then we can mask out the err dimension
        data = [self.predict(p, return_err=True) for p in pose]
        data = np.asarray(data)
        # if it is 1D array, we need to convert to 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        preds = data[:, 0]
        if return_err:
            errs = data[:, 1]
        # return a scalar float value if we only have one input
        if np.all(np.array(preds.shape) == 1):
            preds = preds.item()
            if return_err:
                errs = errs.item()
        else:
            # flatten the array if we have multiple inputs
            preds = preds.flatten()
            if return_err:
                errs = errs.flatten()
        if return_err:
            return preds, errs
        else:
            return preds


class SchnetInference(StructuralInference):
    """
    Inference class for SchNet model.
    """

    model_type: ClassVar[ModelType.schnet] = ModelType.schnet


class E3nnInference(StructuralInference):
    """
    Inference class for E3NN model.
    """

    model_type: ClassVar[ModelType.e3nn] = ModelType.e3nn

    def predict_from_structure_file(self, pose, return_err=False):
        """
        Overload the base class method to pass for_e3nn=True.
        """
        return super().predict_from_structure_file(
            pose, for_e3nn=True, return_err=return_err
        )

    def predict_from_oemol(self, pose, return_err=False):
        """
        Overload the base class method to pass for_e3nn=True.
        """
        return super().predict_from_oemol(pose, for_e3nn=True, return_err=return_err)


_inferences_classes_meta = [
    InferenceBase,
    GATInference,
    StructuralInference,
    SchnetInference,
    E3nnInference,
]


def get_inference_cls_from_model_type(model_type: ModelType):
    instantiable_classes = [
        m for m in _inferences_classes_meta if m.model_type != ModelType.INVALID
    ]
    model_class = [m for m in instantiable_classes if m.model_type == model_type]
    if len(model_class) != 1:
        raise Exception("Somehow got multiple models")
    return model_class[0]
