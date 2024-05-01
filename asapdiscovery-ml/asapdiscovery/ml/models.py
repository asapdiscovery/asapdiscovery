import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union  # noqa: F401
from urllib.parse import urljoin

import mtenn
import pooch
import yaml
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.ml.pretrained_models import asap_models_yaml
from mtenn.config import ModelType
from pydantic import BaseModel, Field, HttpUrl, validator
from semver import Version


class MLModelBase(BaseModel):
    """
    Base model class for ML models
    """

    class Config:
        validate_assignment = True

        # Add custom encoders for semver Versions
        json_encoders = {Version: lambda v: str(v)}

        # Allow arbitrary types so that pydantic will accept Versions
        arbitrary_types_allowed = True

    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    last_updated: date = Field(..., description="Last updated datetime")
    targets: set[TargetTags] = Field(..., description="Biological targets of the model")
    mtenn_lower_pin: Version | None = Field(
        None, description="Lower bound on compatible mtenn versions (inclusive)."
    )
    mtenn_upper_pin: Version | None = Field(
        None, description="Upper bound on compatible mtenn versions (exclusive)."
    )

    @validator("mtenn_lower_pin", "mtenn_upper_pin", pre=True)
    def cast_versions(cls, v):
        """
        Cast SemVer version strings to Version objects.
        """
        if v is None:
            return None
        elif isinstance(v, Version):
            return v
        else:
            return Version.parse(v)

    def check_mtenn_version(self):
        """
        Convenience function for checking the installed mtenn version is compatible with
        the versions specified by the pins.
        """

        try:
            cur_version = Version.parse(mtenn.__version__)
        except AttributeError:
            warnings.warn(
                "No mtenn version found. Assuming compatibility, but note "
                "that this may be incorrect."
            )
            return True

        # If no lower/upper pin has been set, set temp values here that will pass
        if self.mtenn_lower_pin is None:
            low_pin = Version.parse("0.0.0")
        else:
            low_pin = self.mtenn_lower_pin
        if self.mtenn_upper_pin is None:
            # Bumping the patch from current version will make sure current version is
            #  included in exclusive comparison
            upper_pin = cur_version.bump_patch()
        else:
            upper_pin = self.mtenn_lower_pin

        return low_pin <= cur_version < upper_pin


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
                    url=weights_url,
                    known_hash=self.weights_sha256hash,
                    path=local_dir,
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
            targets=self.targets,
            local_dir=Path(local_dir) if local_dir else None,
            mtenn_lower_pin=self.mtenn_lower_pin,
            mtenn_upper_pin=self.mtenn_upper_pin,
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

    models: dict[str, MLModelSpec] = Field(
        ..., description="Models in the model registry, keyed by name"
    )

    def get_models_for_target_and_type(
        self, target: TargetTags, type: ModelType
    ) -> list[MLModelSpec]:
        """
        Get available model specs for a target and type.

        Parameters
        ----------
        target : TargetTags
            Target to get models for
        type : ModelType
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
        if type not in ModelType.get_values():
            raise ValueError(
                f"Model type {type} not valid, must be one of {ModelType.get_values()}"
            )

        return [
            model
            for model in self.models.values()
            if target in model.targets and model.type == type
        ]

    def get_models_for_target(self, target: TargetTags) -> list[MLModelSpec]:
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
        return [model for model in self.models.values() if target in model.targets]
    
    def get_targets_with_models(self) -> List[TargetTags]:
        """
        Get all targets with models

        Returns
        -------
        List[TargetTags]
            List of targets with models
        """
        return list({target.value for model in self.models.values() for target in model.targets})

    def get_latest_model_for_target_and_type(
        self, target: TargetTags, type: ModelType
    ) -> MLModelSpec:
        """
        Get latest model spec for a target

        Parameters
        ----------
        target : TargetTags
            Target to get model for
        type : ModelType
            Type of model to get

        Returns
        -------
        MLModelSpec
            Latest model spec
        """
        models = self.get_models_for_target_and_type(target, type)
        if len(models) == 0:
            warnings.warn(f"No models available for target {target} and type {type}")
            return None
        else:
            return max(models, key=lambda model: model.last_updated)

    def get_model(self, name: str) -> MLModelSpec:
        """
        Get model by name

        Parameters
        ----------
        name : str

        Returns
        -------
        MLModelSpec
            Model spec
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found in model registry")
        return self.models[name]

    def get_implemented_model_types(self):
        """
        Get list of implemented model types

        Returns
        -------
        List[str]
            List of implemented model types
        """
        model_types = {model.type.value for model in self.models.values()}
        return list(model_types)

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
                config_resource=(
                    model_data["config"]["resource"] if has_config else None
                ),
                config_sha256hash=(
                    model_data["config"]["sha256hash"] if has_config else None
                ),
                last_updated=model_data["last_updated"],
                targets=set(model_data["targets"]),
                mtenn_lower_pin=(
                    model_data["mtenn_lower_pin"]
                    if "mtenn_lower_pin" in model_data
                    else None
                ),
                mtenn_upper_pin=(
                    model_data["mtenn_upper_pin"]
                    if "mtenn_upper_pin" in model_data
                    else None
                ),
            )

        return cls(models=models)


# default model registry for all ASAP models
ASAPMLModelRegistry = MLModelRegistry.from_yaml(asap_models_yaml)
