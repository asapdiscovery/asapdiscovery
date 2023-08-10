from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional  # noqa: F401
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum
from datetime import date

from urllib.parse import urljoin

from asapdiscovery.ml.pretrained_models import all_models
from asapdiscovery.data.postera.manifold_data_validation import TargetTags

import pooch
import yaml


class MLModelType(str, Enum):
    GAT = "GAT"
    schnet = "schnet"
    e3nn = "e3nn"
    INVALID = "INVALID"


class MLModelBase(BaseModel):
    name: str = Field(..., description="Model name")
    type: MLModelType = Field(..., description="Model type")
    last_updated: date = Field(..., description="Last updated datetime")
    target: TargetTags = Field(..., description="Biological target of the model")


class MLModelSpec(MLModelBase):
    """
    Model spec for a model stored on S3
    """

    base_url: HttpUrl = Field(..., description="Base url for model files")
    weights_resource: str = Field(..., description="Weights file resource name")
    weights_sha256hash: str = Field(..., description="Weights file sha256 hash")
    config_resource: Optional[str] = Field(None, description="Config resource name")
    config_sha256hash: Optional[str] = Field(None, description="Config sha256 hash")

    def pull(self) -> "LocalMLModelSpec":
        """
        Pull model from S3
        """

        weights_url = urljoin(self.base_url, self.weights_resource)
        try:
            weights_file = Path(
                pooch.retrieve(url=weights_url, known_hash=self.weights_sha256hash)
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
                    pooch.retrieve(url=config_url, known_hash=self.config_sha256hash)
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
    Model spec for a model instantiated locally
    """

    weights_file: Path = Field(..., description="Weights file path")
    config_file: Optional[Path] = Field(..., description="Config path")


class MLModelRegistry(BaseModel):
    """Model registry."""

    models: Dict[str, MLModelSpec] = Field(..., description="Model registry")

    def get_models_for_target_and_type(
        self, target: TargetTags, type: MLModelType
    ) -> List[MLModelSpec]:
        """Get available model specs for a target and type."""
        return [
            model
            for model in self.models.values()
            if model.target == target and model.type == type
        ]

    def get_latest_model_for_target_and_type(
        self, target: TargetTags, type: MLModelType
    ) -> MLModelSpec:
        """Get latest model spec for a target."""
        models = self.get_models_for_target_and_type(target, type)
        latest_name = max(models, key=lambda model: model.last_updated)
        return self.models[latest_name]

    def get_model(name: str) -> MLModelSpec:
        """Get model for a given name."""
        return self.models[name]

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> "MLModelRegistry":
        """Make model registry from yaml spec file"""
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


ASAPMLModelRegistry = MLModelRegistry.from_yaml(all_models)
