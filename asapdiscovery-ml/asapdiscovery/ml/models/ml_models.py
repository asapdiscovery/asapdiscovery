from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional  # noqa: F401
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum
from datetime import date

from urllib.parse import urljoin

from asapdiscovery.ml.pretrained_models import asap_models_yaml
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

import pooch
import yaml


class MLModelType(str, Enum):
    """
    Enum for model types

    GAT: Graph Attention Network
    schnet: SchNet
    e3nn: E(3)-equivariant neural network
    INVALID: Invalid model type to catch instantiation errors
    """

    GAT = "GAT"
    schnet = "schnet"
    e3nn = "e3nn"
    INVALID = "INVALID"


class MLModelBase(BaseModel):
    """
    Base model class for ML models
    """

    name: str = Field(..., description="Model name")
    type: MLModelType = Field(..., description="Model type")
    last_updated: date = Field(..., description="Last updated datetime")
    target: TargetTags = Field(..., description="Biological target of the model")


class MLModelSpec(MLModelBase):
    """
    Model spec for a model stored at a remote url
    """

    base_url: HttpUrl = Field(..., description="Base url for model files")
    weights_resource: str = Field(..., description="Weights file resource name")
    weights_sha256hash: str = Field(..., description="Weights file sha256 hash")
    config_resource: Optional[str] = Field(None, description="Config resource name")
    config_sha256hash: Optional[str] = Field(None, description="Config sha256 hash")

    def pull(self, local_dir: Union[Path, str] = None) -> "LocalMLModelSpec":
        """
        Pull model from S3

        Parameters
        ----------
        local_dir : Union[Path, str], optional
            Local directory to store model files, by default None, meaning they will be stored in pooch.os_cache

        Returns
        -------
        LocalMLModelSpec
            Local model spec
        """

        weights_url = urljoin(self.base_url, self.weights_resource)
        try:
            weights_file = Path(
                pooch.retrieve(
                    url=weights_url, known_hash=self.weights_sha256hash, path=local_dir
                )
            )
        except Exception as e:
            raise ValueError(
                f"Model {self.name} weights file {self.weights_resource} from {weights_url} download failed, please check your yaml spec file for errors."
            ) from e

        # fetch config
        if self.config_resource:
            config_url = urljoin(self.base_url, self.config_resource)
            try:
                config_file = Path(
                    pooch.retrieve(
                        url=config_url,
                        known_hash=self.config_sha256hash,
                        path=local_dir,
                    )
                )
            except Exception as e:
                raise ValueError(
                    f"Model {self.name} config file {self.config_resource} from {config_url} download failed, please check your yaml spec file for errors."
                ) from e

        return LocalMLModelSpec(
            name=self.name,
            type=self.type,
            weights_file=Path(weights_file),
            config_file=Path(config_file) if self.config_resource else None,
            last_updated=self.last_updated,
            target=self.target,
        )


class LocalMLModelSpec(MLModelBase):
    """
    Model spec for a model instantiated locally, containing file paths to model files
    """

    weights_file: Path = Field(..., description="Weights file path")
    config_file: Optional[Path] = Field(..., description="Optional config file path")
    local_dir: Optional[Path] = Field(
        None,
        description="Local directory for model files, otherwise defaults to pooch.os_cache",
    )


class MLModelRegistry(BaseModel):
    """
    Model registry for ML models stored remotely, read from a yaml spec file most of the time

    """

    models: Dict[str, MLModelSpec] = Field(
        ..., description="Models in the model registry, keyed by name"
    )

    def get_models_for_target_and_type(
        self, target: TargetTags, type: MLModelType
    ) -> List[MLModelSpec]:
        """
        Get available model specs for a target and type.

        Parameters
        ----------
        target : TargetTags
            Target to get models for
        type : MLModelType
            Type of model to get

        Returns
        -------
        List[MLModelSpec]
            List of model specs
        """
        if target not in TargetTags.get_values():
            raise ValueError(
                f"Target {target} not valid, must be one of {TargetTags.get_values()}"
            )

        return [
            model
            for model in self.models.values()
            if model.target == target and model.type == type
        ]

    def get_models_for_target(self, target: TargetTags) -> List[MLModelSpec]:
        """
        Get available model specs for a target

        Parameters
        ----------
        target : TargetTags
            Target to get models for

        Returns
        -------
        List[MLModelSpec]
            List of model specs
        """
        if target not in TargetTags.get_values():
            raise ValueError(
                f"Target {target} not valid, must be one of {TargetTags.get_values()}"
            )
        return [model for model in self.models.values() if model.target == target]

    def get_latest_model_for_target_and_type(
        self, target: TargetTags, type: MLModelType
    ) -> MLModelSpec:
        """
        Get latest model spec for a target

        Parameters
        ----------
        target : TargetTags
            Target to get model for
        type : MLModelType
            Type of model to get

        Returns
        -------
        MLModelSpec
            Latest model spec
        """
        models = self.get_models_for_target_and_type(target, type)
        return max(models, key=lambda model: model.last_updated)

    def get_model(self, name: str) -> MLModelSpec:
        """
        Get model for by name

        Parameters
        ----------
        name : str

        Returns
        -------
        MLModelSpec
            Model spec
        """
        return self.models[name]

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> "MLModelRegistry":
        """
        Make model registry from yaml spec file

        Parameters
        ----------
        yaml_file : Union[str, Path]

        Returns
        -------
        MLModelRegistry
            Model registry
        """
        if not Path(yaml_file).exists():
            raise FileNotFoundError(f"Yaml spec file {yaml_file} does not exist")

        with open(yaml_file) as f:
            spec = yaml.safe_load(f)

        models = {}
        for model in spec:
            model_data = spec[model]
            has_config = "config" in model_data
            models[model] = MLModelSpec(
                name=model,
                type=model_data["type"],
                base_url=model_data["base_url"],
                weights_resource=model_data["weights"]["resource"],
                weights_sha256hash=model_data["weights"]["sha256hash"],
                config_resource=model_data["config"]["resource"]
                if has_config
                else None,
                config_sha256hash=model_data["config"]["sha256hash"]
                if has_config
                else None,
                last_updated=model_data["last_updated"],
                target=model_data["target"],
            )

        return cls(models=models)


# default model registry for all ASAP models
ASAPMLModelRegistry = MLModelRegistry.from_yaml(asap_models_yaml)
