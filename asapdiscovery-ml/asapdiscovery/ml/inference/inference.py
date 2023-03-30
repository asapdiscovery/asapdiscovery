import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import dgl

# static import of models from base yaml here
from asapdiscovery.ml.pretrained_models import all_models
from asapdiscovery.ml.utils import build_model, load_weights
from asapdiscovery.ml.weights import ModelSpec, fetch_file, fetch_model_from_spec


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
        model, _ = build_model(model_type, **kwargs)
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
        device: str = "cpu",
        build_model_kwargs: Optional[dict] = None,
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

        """
        with torch.no_grad():
            output_tensor = self.model(g, g.ndata["h"])
            output_tensor = torch.reshape(output_tensor, (-1, 1))
            return output_tensor.cpu().numpy()
