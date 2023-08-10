import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, ClassVar  # noqa: F401

import dgl
import numpy as np
import torch
from asapdiscovery.ml.dataset import DockedDataset, GraphInferenceDataset

from pydantic import BaseModel, Field, validator

# static import of models from base yaml here
from asapdiscovery.ml.utils import build_model, load_weights
from asapdiscovery.ml.models.ml_models import (
    MLModelRegistry,
    ASAPMLModelRegistry,
    MLModelType,
    MLModelSpec,
    LocalMLModelSpec,
)
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

from dgllife.utils import CanonicalAtomFeaturizer


class InferenceBase(BaseModel):
    class Config:
        validate_assignment = True
        allow_mutation = False
        arbitrary_types_allowed = True
        allow_extra = False

    target: TargetTags = Field(..., description="Target to predict for")
    model_type: ClassVar[MLModelType.INVALID] = Field(MLModelType.INVALID)
    model_name: str = Field(..., description="Name of model to use")
    model_spec: Optional[MLModelSpec] = Field(
        ..., description="Model spec used to create Model to use"
    )
    local_model_spec: LocalMLModelSpec = Field(
        ..., description="Local model spec used to create Model to use"
    )
    build_model_kwargs: Optional[Dict] = Field(
        ..., description="Keyword arguments to pass to build_model function"
    )
    device: str = Field("cpu", description="Device to use for inference")
    model: torch.nn.Module = Field(..., description="PyTorch model")

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
        return cls.from_MLModelSpec(model_spec, **kwargs)

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
        return cls.from_MLModelSpec(model_spec, **kwargs)

    @classmethod
    def from_MLModelSpec(
        cls,
        model_spec: MLModelSpec,
        device: str = "cpu",
        local_dir: Optional[Union[str, Path]] = "_weights",
    ) -> "_InferenceBase":
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
            model_components, device=device, model_spec=model_spec
        )

    @classmethod
    def from_local_model_spec(
        cls,
        local_model_spec: LocalMLModelSpec,
        device: str = "cpu",
        model_spec: Optional[MLModelSpec] = None,
    ) -> "_InferenceBase":
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
        model = build_model(local_model_spec.type, config=local_model_spec.config_file)
        model = load_weights(
            model, local_model_spec.weights_file, check_compatibility=True
        )
        model.eval()
        return cls(
            target=local_model_spec.target,
            model_type=local_model_spec.type,
            model_name=local_model_spec.name,
            model_spec=model_spec,
            local_model_spec=local_model_spec,
            build_model_kwargs={},
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
    model_type: ClassVar[MLModelType.GAT] = Field(MLModelType.GAT)

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

    model_type: ClassVar[MLModelType.INVALID] = Field(MLModelType.INVALID)

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

    model_type: ClassVar[MLModelType.schnet] = Field(MLModelType.schnet)
    build_model_kwargs: Dict[str, str] = Field({"pred_r": "pIC50"})


class E3nnInference(StructuralInference):
    """
    Inference class for E3NN model.
    """

    model_type: ClassVar[MLModelType.e3nn] = Field(MLModelType.e3nn)


# class InferenceBase:
#     """
#     Inference base class for PyTorch models in asapdiscovery.

#     Parameters
#     ----------
#     model_name : str
#         Name of model to use.
#     model_type : str
#         Type of model to use.
#     model_spec : Path, default=None
#         The path to the model spec yaml file. If not specified, the default
#         asapdiscovery.ml models.yaml file will be used.
#     weights_local_dir: Path, default="./_weights/"
#         The path to the local directory to store the weights. If not specified,
#         will use the default `_weights` directory in the current working
#         directory.
#     build_model_kwargs : Optional[Dict], default=None
#         Keyword arguments to pass to build_model function.
#     device : str, default='cpu'
#         Device to use for inference.


#     Methods
#     -------
#     predict
#         Predict on a batch of data.
#     build_model
#         Build model from arguments

#     """

#     model_type = MLModelType.INVALID

#     def __init__(
#         self,
#         target: TargetTags,
#         model_name: Optional[str] = None,
#         model_registry: MLModelRegistry = ASAPMLModelRegistry,
#         weights_local_dir: Union[Path, str] = Path("./_weights/"),
#         build_model_kwargs: Optional[dict] = None,
#         device: str = "cpu",
#     ):
#         logging.info(f"initializing {self.__class__.__name__} class")

#         self.device = device
#         logging.info(f"using device {self.device}")

#         if self.model_type == MLModelType.INVALID:
#             raise ValueError(
#                 "Model type is invalid, do not instantiate one of the base classes directly, inherit from it."
#             )
#         if target not in TargetTags.get_values():
#             raise ValueError(
#                 f"Invalid target, must be one of {TargetTags.get_values()}"
#             )

#         self.target = target
#         self.model_registry = model_registry
#         self.weights_local_dir = weights_local_dir

#         if model_name:
#             if model_name not in self.model_registry.get_models_for_target_and_type(
#                 self.target, self.model_type
#             ):
#                 raise ValueError(
#                     f"Model {model_name} not found for target {self.target} and type {self.model_type}"
#                 )
#             self.model_spec = self.model_registry.get_model(model_name)
#         else:
#             self.model_spec = self.model_registry.get_latest_model_for_target_and_type(
#                 self.model_type, self.target
#             )

#         logging.info(f"found ML model spec {self.model_spec}")

#         # pull the model down
#         self.model_components = self.model_spec.pull()
#         logging.info(f"pulled model components {self.model_components}")

#         # build model kwargs
#         if not build_model_kwargs:
#             build_model_kwargs = {}

#         # add extra kwargs if they exist in class def
#         extra_build_model_kwargs = (
#             self._extra_build_model_kwargs
#             if hasattr(self._extra_build_model_kwargs)
#             else {}
#         )
#         build_model_kwargs = {**build_model_kwargs, **extra_build_model_kwargs}

#         # add config file if it exists
#         if (
#             build_model_kwargs.get("config") is None
#             and self.model_components.config_file
#         ):
#             build_model_kwargs["config"] = self.model_components.config_file

#         # build model, this needs a bit of cleaning up in the function itself.
#         self.model = self.build_model(self.model_components.type, **build_model_kwargs)
#         logging.info(f"built model {self.model}")

#         # load weights
#         self.model = load_weights(
#             self.model, self.model_components.weights_file, check_compatibility=True
#         )
#         logging.info(f"loaded weights {self.model_components.weights}")

#         self.model.eval()
#         logging.info("set model to eval mode")

#     def build_model(self, model_type: str, **kwargs):
#         """can be overloaded in child classes for more complex setups,
#         but most uses should be fine with this, needs to return a
#         torch.nn.Module is only real requirement.

#         Parameters
#         ----------
#         model_type : str
#             Type of model to use.
#         **kwargs
#             Keyword arguments to pass to build_model function.

#         Returns
#         -------
#         model: torch.nn.Module
#             PyTorch model.
#         """
#         model = build_model(model_type, **kwargs)
#         return model

#     def predict(self, input_data):
#         """Predict on data, needs to be overloaded in child classes most of
#         the time

#         Parameters
#         ----------

#         input_data: pytorch.Tensor

#         Returns
#         -------
#         np.ndarray
#             Prediction from model.
#         """
#         # feed in data in whatever format is required by the model
#         with torch.no_grad():
#             input_tensor = torch.tensor(input_data).to(self.device)
#             output_tensor = self.model(input_tensor)
#             return output_tensor.cpu().numpy().ravel()


# # this is just an example of how to use the base class, we may want to specialise this for each model type


# class GATInference(InferenceBase):
#     """
#     Inference class for GAT model.

#     """

#     model_type = MLModelType.GAT

#     def predict(self, g: dgl.DGLGraph):
#         """Predict on a graph, requires a DGLGraph object with the `ndata`
#         attribute `h` containing the node features. This is done by constucting
#         the `GraphDataset` with the node_featurizer=`dgllife.utils.CanonicalAtomFeaturizer()`
#         argument.


#         Parameters
#         ----------
#         g : dgl.DGLGraph
#             DGLGraph object.

#         Returns
#         -------
#         np.ndarray
#             Predictions for each graph.
#         """
#         with torch.no_grad():
#             output_tensor = self.model({"g": g})
#             # we ravel to always get a 1D array
#             return output_tensor.cpu().numpy().ravel()

#     def predict_from_smiles(
#         self, smiles: Union[str, list[str]], **kwargs
#     ) -> Union[np.ndarray, float]:
#         """Predict on a list of SMILES strings, or a single SMILES string.

#         Parameters
#         ----------
#         smiles : Union[str, List[str]]
#             SMILES string or list of SMILES strings.

#         Returns
#         -------
#         np.ndarray or float
#             Predictions for each graph, or a single prediction if only one SMILES string is provided.
#         """
#         if isinstance(smiles, str):
#             smiles = [smiles]

#         gids = GraphInferenceDataset(
#             smiles, node_featurizer=CanonicalAtomFeaturizer(), **kwargs
#         )

#         data = [self.predict(g) for g in gids]
#         data = np.concatenate(np.asarray(data))
#         # return a scalar float value if we only have one input
#         if np.all(np.array(data.shape) == 1):
#             data = data.item()
#         return data


# class StructuralInference(InferenceBase):
#     """
#     Inference class for models that take a structure as input.
#     """

#     model_type = MLModelType.INVALID

#     def predict(self, pose_dict: dict):
#         """Predict on a pose, requires a dictionary with the pose data with
#         the keys: "z", "pos", "lig" with the required tensors in each

#         Parameters
#         ----------
#         pose_dict : dict
#             Dictionary with pose data.

#         Returns
#         -------
#         np.ndarray
#             Predictions for a pose.
#         """
#         with torch.no_grad():
#             output_tensor = self.model(pose_dict)
#             # we ravel to always get a 1D array
#             return output_tensor.cpu().numpy().ravel()

#     def predict_from_structure_file(
#         self, pose: Union[Path, list[Path]]
#     ) -> Union[np.ndarray, float]:
#         """Predict on a list of poses or a single pose.

#         Parameters
#         ----------
#         pose : Union[Path, List[Path]]
#             Path to pose file or list of paths to pose files.

#         Returns
#         -------
#         np.ndarray or float
#             Prediction for poses, or a single prediction if only one pose is provided.
#         """

#         if isinstance(pose, Path):
#             pose = [pose]

#         pose = [DockedDataset._load_structure(p, None) for p in pose]
#         data = [self.predict(p) for p in pose]

#         data = np.concatenate(np.asarray(data))
#         # return a scalar float value if we only have one input
#         if np.all(np.array(data.shape) == 1):
#             data = data.item()
#         return data


# class SchnetInference(StructuralInference):
#     """
#     Inference class for SchNet model.
#     """

#     model_type = "schnet"
#     _extra_build_model_kwargs = {"pred_r": "pIC50"}


# class E3nnInference(StructuralInference):
#     """
#     Inference class for E3NN model.
#     """

#     model_type = "e3nn"
