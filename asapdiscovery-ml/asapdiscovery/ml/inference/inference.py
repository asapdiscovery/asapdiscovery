from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union  # noqa: F401

import dgl
import numpy as np
import torch
from asapdiscovery.data.postera.manifold_data_validation import TargetTags
from asapdiscovery.ml.dataset import DockedDataset, GraphInferenceDataset
from asapdiscovery.ml.models.ml_models import (
    ASAPMLModelRegistry,
    LocalMLModelSpec,
    MLModelRegistry,
    MLModelSpec,
    MLModelType,
)

# static import of models from base yaml here
from asapdiscovery.ml.utils import build_model, load_weights
from dgllife.utils import CanonicalAtomFeaturizer
from pydantic import BaseModel, Field


class InferenceBase(BaseModel):
    class Config:
        validate_assignment = True
        allow_mutation = False
        arbitrary_types_allowed = True
        allow_extra = False

    targets: set[TargetTags] = Field(
        ..., description="Targets that them model can predict for"
    )
    model_type: ClassVar[MLModelType.INVALID] = MLModelType.INVALID
    model_name: str = Field(..., description="Name of model to use")
    model_spec: Optional[MLModelSpec] = Field(
        ..., description="Model spec used to create Model to use"
    )
    local_model_spec: LocalMLModelSpec = Field(
        ..., description="Local model spec used to create Model to use"
    )
    device: str = Field("cpu", description="Device to use for inference")
    model: Optional[torch.nn.Module] = Field(..., description="PyTorch model")

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
        local_model_spec: LocalMLModelSpec,
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
        model = build_model(
            local_model_spec.type,
            config=local_model_spec.config_file,
            **build_model_kwargs,
        )
        model = load_weights(
            model, local_model_spec.weights_file, check_compatibility=True
        )
        model.eval()

        return cls(
            targets=local_model_spec.targets,
            model_type=local_model_spec.type,
            model_name=local_model_spec.name,
            model_spec=model_spec,
            local_model_spec=local_model_spec,
            device=device,
            model=model,
        )

    def predict(self, input_data):
        """Predict on data, needs to be overloaded in child classes most of
        the time

        Parameters
        ----------

        input_data: pytorch.Tensor

        Returns
        -------
        np.ndarray
            Prediction from model.
        """
        # feed in data in whatever format is required by the model
        with torch.no_grad():
            input_tensor = torch.tensor(input_data).to(self.device)
            output_tensor = self.model(input_tensor)
            return output_tensor.cpu().numpy().ravel()


class GATInference(InferenceBase):
    model_type: ClassVar[MLModelType.GAT] = MLModelType.GAT

    def predict(self, g: dgl.DGLGraph):
        """Predict on a graph, requires a DGLGraph object with the `ndata`
        attribute `h` containing the node features. This is done by constucting
        the `GraphDataset` with the node_featurizer=`dgllife.utils.CanonicalAtomFeaturizer()`
        argument.


        Parameters
        ----------
        g : dgl.DGLGraph
            DGLGraph object.

        Returns
        -------
        np.ndarray
            Predictions for each graph.
        """
        with torch.no_grad():
            output_tensor = self.model({"g": g})
            # we ravel to always get a 1D array
            return output_tensor.cpu().numpy().ravel()

    def predict_from_smiles(
        self, smiles: Union[str, list[str]], **kwargs
    ) -> Union[np.ndarray, float]:
        """Predict on a list of SMILES strings, or a single SMILES string.

        Parameters
        ----------
        smiles : Union[str, List[str]]
            SMILES string or list of SMILES strings.

        Returns
        -------
        np.ndarray or float
            Predictions for each graph, or a single prediction if only one SMILES string is provided.
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        gids = GraphInferenceDataset(
            smiles, node_featurizer=CanonicalAtomFeaturizer(), **kwargs
        )

        data = [self.predict(g) for g in gids]
        data = np.concatenate(np.asarray(data))
        # return a scalar float value if we only have one input
        if np.all(np.array(data.shape) == 1):
            data = data.item()
        return data


class StructuralInference(InferenceBase):
    """
    Inference class for models that take a structure as input.
    """

    model_type: ClassVar[MLModelType.INVALID] = MLModelType.INVALID

    def predict(self, pose_dict: dict):
        """Predict on a pose, requires a dictionary with the pose data with
        the keys: "z", "pos", "lig" with the required tensors in each

        Parameters
        ----------
        pose_dict : dict
            Dictionary with pose data.

        Returns
        -------
        np.ndarray
            Predictions for a pose.
        """
        with torch.no_grad():
            output_tensor = self.model(pose_dict)
            # we ravel to always get a 1D array
            return output_tensor.cpu().numpy().ravel()

    def predict_from_structure_file(
        self, pose: Union[Path, list[Path]]
    ) -> Union[np.ndarray, float]:
        """Predict on a list of poses or a single pose.

        Parameters
        ----------
        pose : Union[Path, List[Path]]
            Path to pose file or list of paths to pose files.

        Returns
        -------
        np.ndarray or float
            Prediction for poses, or a single prediction if only one pose is provided.
        """

        if isinstance(pose, Path):
            pose = [pose]

        pose = [DockedDataset._load_structure(p, None) for p in pose]
        data = [self.predict(p) for p in pose]

        data = np.concatenate(np.asarray(data))
        # return a scalar float value if we only have one input
        if np.all(np.array(data.shape) == 1):
            data = data.item()
        return data


class SchnetInference(StructuralInference):
    """
    Inference class for SchNet model.
    """

    model_type: ClassVar[MLModelType.schnet] = MLModelType.schnet


class E3nnInference(StructuralInference):
    """
    Inference class for E3NN model.
    """

    model_type: ClassVar[MLModelType.e3nn] = MLModelType.e3nn
