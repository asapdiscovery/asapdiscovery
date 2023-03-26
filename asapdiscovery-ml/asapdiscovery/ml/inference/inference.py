from typing import Optional
import logging

import torch

# static import of models from base yaml here
from asapdiscovery.ml.weights import (
    fetch_weights_from_spec,
    fetch_weights,
    all_models_spec,
)

from asapdiscovery.ml.utils import load_weights, build_model


class InferenceBase:
    """
    Inference base class for PyTorch models in asapdiscovery.

    Parameters
    ----------
    model_name : str
        Name of model to use.
    model_type : str
        Type of model to use.
    from_spec : bool, default=True
        Whether to fetch weights from asapdiscovery.ml spec file or from local file.
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
        from_spec: bool = True,
        build_model_kwargs: Optional[Dict] = None,
        device: str = "cpu",
    ):
        logging.info(f"initializing {self.__class__.__name__} class")

        self.device = device
        logging.info(f"using device {self.device}")

        self.model_name = model_name
        self.model_type = model_type
        logging.info(f"using model {self.model_name} of type {self.model_type}")

        # load model weights or fetch them
        if from_spec:
            logging.info("using weights from asapdiscovery.ml models.yaml spec file")
            weights, types = fetch_weights_from_spec(all_models_spec, model_name)
            if types[model_name] != self.model_type:
                raise ValueError(
                    f"Model type {types[model_name]} does not match {self.model_type}"
                )
            self.weights = weights[model_name]
        else:
            logging.info("using weights from specified local file or spec")
            if self.model_name.split(".")[-1] in ["yaml", "yml"]:
                logging.info("local yaml file specified, fetching weights from spec")
                weights, types = fetch_weights_from_spec(self.model_name, model_name)
                if types[model_name] != self.model_type:
                    raise ValueError(
                        f"Model type {types[model_name]} does not match {self.model_type}"
                    )
                self.weights = weights[model_name]
            else:
                logging.info("local weights file specified, fetching weights from file")
                self.weights = fetch_weights(model_name)

        logging.info(f"found weights {self.weights}")

        # build model, this needs a bit of cleaning up in the function itself.
        self.model, _ = build_model(model_type, **build_model_kwargs)
        logging.info(f"built model {self.model}")

        # load weights
        self.model = load_weights(self.model, self.weights)
        logging.info(f"loaded weights {self.weights}")

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

    def __init__(self, model_name: str, from_spec: bool = True, device: str = "cpu"):
        # build model kwargs specific to the GAT model, alternatively we can allow these to be passed in
        build_model_kwargs = {}
        super().__init__(
            model_name,
            self.model_type,
            from_spec=from_spec,
            build_model_kwargs=build_model_kwargs,
            device=device,
        )
