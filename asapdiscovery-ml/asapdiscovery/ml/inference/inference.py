import logging
from pathlib import Path
from typing import Dict, List, Optional, Union  # noqa: F401

import dgl
import numpy as np
import torch
from asapdiscovery.ml.dataset import DockedDataset, GraphInferenceDataset

# static import of models from base yaml here
from asapdiscovery.ml.utils import build_model, load_weights
from asapdiscovery.ml.weights import MLModelRegistry, DefaultModelRegistry, MLModelType
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

from dgllife.utils import CanonicalAtomFeaturizer


class InferenceBase:
    """
    Inference base class for PyTorch models in asapdiscovery.

    Parameters
    ----------
    model_name : str
        Name of model to use.
    model_type : str
        Type of model to use.
    model_spec : Path, default=None
        The path to the model spec yaml file. If not specified, the default
        asapdiscovery.ml models.yaml file will be used.
    weights_local_dir: Path, default="./_weights/"
        The path to the local directory to store the weights. If not specified,
        will use the default `_weights` directory in the current working
        directory.
    build_model_kwargs : Optional[Dict], default=None
        Keyword arguments to pass to build_model function.
    device : str, default='cpu'
        Device to use for inference.


    Methods
    -------
    predict
        Predict on a batch of data.
    build_model
        Build model from arguments

    """

    model_type = MLModelType.INVALID

    def __init__(
        self,
        target: TargetTags,
        model_name: Optional[str],
        model_registry: MLModelRegistry = DefaultModelRegistry,
        weights_local_dir: Union[Path, str] = Path("./_weights/"),
        build_model_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ):
        logging.info(f"initializing {self.__class__.__name__} class")

        self.device = device
        logging.info(f"using device {self.device}")

        if self.model_type == MLModelType.INVALID:
            raise ValueError(
                "Model type is invalid, do not instantiate one of the base classes directly, inherit from it."
            )

        self.target = target
        self.model_registry = model_registry
        self.weights_local_dir = weights_local_dir

        if self.model_name:
            if model_name not in self.model_registry.get_models_for_target_and_type(
                self.target, self.model_type
            ):
                raise ValueError(
                    f"Model {model_name} not found for target {self.target} and type {self.model_type}"
                )
            self.model_spec = self.model_registry.get_model(model_name)
        else:
            self.model_spec = self.model_registry.get_latest_model_for_target(
                self.model_type, self.target
            )

        logging.info(f"found ML model spec {self.model_spec}")

        # pull the model down
        self.model_components = self.model_spec.pull()
        logging.info(f"pulled model components {self.model_components}")

        # build model kwargs
        if not build_model_kwargs:
            build_model_kwargs = {}
            build_model_kwargs["config"] = self.model_components.config_file

        # otherwise just roll with what we have

        # build model, this needs a bit of cleaning up in the function itself.
        self.model = self.build_model(self.model_components.type, **build_model_kwargs)
        logging.info(f"built model {self.model}")

        # load weights
        self.model = load_weights(
            self.model, self.model_components.weights_file, check_compatibility=True
        )
        logging.info(f"loaded weights {self.model_components.weights}")

        self.model.eval()
        logging.info("set model to eval mode")

    def build_model(self, model_type: str, **kwargs):
        """can be overloaded in child classes for more complex setups,
        but most uses should be fine with this, needs to return a
        torch.nn.Module is only real requirement.

        Parameters
        ----------
        model_type : str
            Type of model to use.
        **kwargs
            Keyword arguments to pass to build_model function.

        Returns
        -------
        model: torch.nn.Module
            PyTorch model.
        """
        model = build_model(model_type, **kwargs)
        return model

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


# this is just an example of how to use the base class, we may want to specialise this for each model type


class GATInference(InferenceBase):
    """
    Inference class for GAT model.

    """

    model_type = MLModelType.GAT

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

    model_type = MLModelType.INVALID

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

    model_type = "schnet"

    def __init__(
        self,
        model_name: str,
        weights_local_dir: Union[Path, str] = Path("./_weights/"),
        build_model_kwargs: Optional[dict] = None,
        pIC50_units=True,
        device: str = "cpu",
    ):
        if pIC50_units:
            if build_model_kwargs:
                build_model_kwargs = {"pred_r": "pIC50"} | build_model_kwargs
            else:
                build_model_kwargs = {"pred_r": "pIC50"}

        super().__init__(
            model_name,
            weights_local_dir=weights_local_dir,
            build_model_kwargs=build_model_kwargs,
            device=device,
        )


class E3nnInference(StructuralInference):
    """
    Inference class for E3NN model.
    """

    model_type = "e3nn"
