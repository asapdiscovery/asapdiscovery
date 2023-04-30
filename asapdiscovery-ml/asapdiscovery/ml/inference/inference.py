import logging
from pathlib import Path
from typing import Dict, List, Optional, Union  # noqa: F401

import dgl
import numpy as np
import torch
from asapdiscovery.ml.dataset import GraphInferenceDataset

# static import of models from base yaml here
from asapdiscovery.ml.pretrained_models import all_models
from asapdiscovery.ml.utils import build_model, load_weights
from asapdiscovery.ml.weights import fetch_model_from_spec
from asapdiscovery.ml.dataset import DockedDataset
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

    def __init__(
        self,
        model_name: str,
        model_type: str,
        model_spec: Path = None,
        build_model_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ):
        logging.info(f"initializing {self.__class__.__name__} class")

        self.device = device
        logging.info(f"using device {self.device}")

        self.model_name = model_name
        self.model_type = model_type
        self.model_spec = model_spec

        self.model_components = None

        logging.info(
            f"using model {self.model_name} of type {self.model_type} from spec {self.model_spec}"
        )

        # load model weights or fetch them
        if not self.model_spec:
            logging.info(
                " no model spec specified, using spec from asapdiscovery.ml models.yaml spec file"
            )
            self.model_spec = all_models
        else:
            logging.info("local yaml file specified, fetching weights from spec")
            if not self.model_spec.split(".")[-1] in ["yaml", "yml"]:
                raise ValueError(
                    f"Model spec file {self.model_spec} is not a yaml file"
                )

        self.model_components = fetch_model_from_spec(self.model_spec, model_name)[
            model_name
        ]
        if self.model_components.type != self.model_type:
            raise ValueError(
                f"Model type {self.model_components.type} does not match {self.model_type}"
            )

        logging.info(f"found weights {self.model_components.weights}")

        # build model kwargs
        if not build_model_kwargs:
            build_model_kwargs = {}
            build_model_kwargs["config"] = self.model_components.config

        # otherwise just roll with what we have

        # build model, this needs a bit of cleaning up in the function itself.
        self.model = self.build_model(self.model_components.type, **build_model_kwargs)
        logging.info(f"built model {self.model}")

        # load weights
        self.model = load_weights(
            self.model, self.model_components.weights, check_compatibility=True
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
        # feed in data in whatever format is required by the model

        with torch.no_grad():
            input_tensor = torch.tensor(input_data).to(self.device)
            output_tensor = self.model(input_tensor)
            return output_tensor.cpu().numpy()


# this is just an example of how to use the base class, we may want to specialise this for each model type
# eg have GAT2DInference, GAT3DInference, etc.


class GATInference(InferenceBase):
    """
    Inference class for GAT model.

    """

    model_type = "GAT"

    def __init__(
        self,
        model_name: str,
        model_spec: Optional[Path] = None,
        build_model_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ):
        super().__init__(
            model_name,
            self.model_type,
            model_spec,
            build_model_kwargs=build_model_kwargs,
            device=device,
        )

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
        if len(data) == 1:
            data = data[0]
        return data


class StructuralInference(InferenceBase):
    """
    Inference class for models that take a structure as input.
    """

    def __init__(
        self,
        model_name: str,
        model_spec: Optional[Path] = None,
        build_model_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ):
        super().__init__(
            model_name,
            self.model_type,
            model_spec,
            build_model_kwargs=build_model_kwargs,
            device=device,
        )

    def predict_from_structure_file(
        self, pose: Union[Path, List[Path]]
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

        pose = [
            DockedDataset._load_structure(
                p,
            )
            for p in pose
        ]
        data = [self.predict(p) for p in pose]

        data = np.concatenate(np.asarray(data))
        # return a scalar float value if we only have one input
        if len(data) == 1:
            data = data[0]

        return data

    def predict_from_pose(
        self, pose: Union[Dict, List[Dict]]
    ) -> Union[np.ndarray, float]:
        """
        Predict on a list of poses or a single pose as generated by a
        DockedDataset (dict of the form {"z", "pos", "lig", "compound"}).

        Parameters
        ----------
        pose : Union[Dict, List[Dict]]
            Pose as generated by a DockedDataset.

        Returns
        -------
        np.ndarray or float
            Prediction for poses, or a single prediction if only one pose is provided.
        """

        if isinstance(pose, dict):
            pose = [pose]

        data = [self.predict(p) for p in pose]

        data = np.concatenate(np.asarray(data))
        # return a scalar float value if we only have one input
        if len(data) == 1:
            data = data[0]
        return data


class SchnetInference(StructuralInference):
    """
    Inference class for SchNet model.
    """

    model_type = "schnet"


class E3nnInference(StructuralInference):
    """
    Inference class for E3NN model.
    """

    model_type = "e3nn"
