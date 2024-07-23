import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union  # noqa: F401
from urllib.parse import urljoin

import mtenn
import pooch
import requests
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
    # class variable ensemble = False

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


class MLModelSpecBase(MLModelBase):
    """Base class for model specs"""

    ensemble: bool = False


class MLModelSpec(MLModelSpecBase):
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


class EnsembleMLModelSpec(MLModelSpecBase):
    models: list[MLModelSpec] = Field(
        ..., description="List of model specs for ensemble models"
    )
    ensemble: bool = True

    @validator("models")
    @classmethod
    def check_all_types(cls, models):
        """
        Check that all models in the ensemble are of the same type
        """
        if len({model.type for model in models}) > 1:
            raise ValueError("All models in an ensemble must be of the same type")
        return models

    @validator("models")
    @classmethod
    def check_all_mtenn_versions(cls, models):
        """
        Check that all models in the ensemble are compatible with the same mtenn version
        """
        if len({model.mtenn_lower_pin for model in models}) > 1:
            raise ValueError(
                "All models in an ensemble must have the same mtenn_lower_pin"
            )
        if len({model.mtenn_upper_pin for model in models}) > 1:
            raise ValueError(
                "All models in an ensemble must have the same mtenn_upper_pin"
            )
        return models

    @property
    def ensemble_size(self):
        return len(self.models)

    def pull(self, local_dir: Union[Path, str] = None) -> "LocalEnsembleMLModelSpec":
        """
        Pull ensemble model from S3

        Parameters
        ----------
        local_dir : Union[Path, str], optional
            Local directory to store model files, by default None, meaning they will be stored in pooch.os_cache

        Returns
        -------
        List[LocalMLModelSpec]
            List of local model specs
        """
        return LocalEnsembleMLModelSpec(
            models=[model.pull(local_dir) for model in self.models],
            **self.dict(exclude={"models"}),
        )


def _url_to_yaml(url: str) -> dict:
    # Retrieve the file content from the URL
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    # Convert bytes to string
    try:
        content = response.content.decode("utf-8")
    except:  # noqa: E722
        raise ValueError(f"Failed to decode content from {url}")
    # Load the yaml
    return yaml.safe_load(content)


class RemoteEnsembleHelper(BaseModel):
    """
    Helper class for remote ensemble models

    Parses manifests of the form

    asapdiscovery-GAT-ensemble-test:
    type: GAT
    base_url: https://asap-discovery-ml-weights.asapdata.org/production/GAT/
    ensemble: True

    weights:
        - asapdiscovery-GAT-X:
            resource: asapdiscovery-GAT-X.th
            sha256hash: 0b1
        - asapdiscovery-GAT-Y:
            resource: asapdiscovery-GAT-Y.th
            sha256hash: 0b2
        - asapdiscovery-GAT-Z:
            resource: asapdiscovery-GAT-Z.th
            sha256hash: 0b3
    config:
        resource: asapdiscovery-SARS-CoV-2-Mpro.json
    targets:
        - SARS-CoV-2-Mpro
        - MERS-CoV-Mpro
    mtenn_lower_pin: "0.5.0"

    """

    manifest_url: HttpUrl = Field(..., description="Remote ensemble model url")

    def to_ensemble_spec(self) -> dict[str,EnsembleMLModelSpec]:
        """
        Convert remote ensemble model to ensemble model spec

        Returns
        -------
        dict[str,EnsembleMLModelSpec]
            Dictionary of ensemble model specs
        """
        try:
            manifest = _url_to_yaml(self.manifest_url)
        except Exception as e:
            warnings.warn(
                f"Failed to load manifest from {self.manifest_url}, skipping. Error: {e}"
            )
            return {}

        ensemble_models = {}

        for model in manifest:
            try:
                model_data = manifest[model]
                models = []
                for submodel in model_data["weights"]:
                    if len(submodel) > 1:
                        raise ValueError("Submodel should have only one key")
                    # get the name of the submodel
                    subname = list(submodel.keys())[0]
                    models.append(
                        MLModelSpec(
                            name=model + "_ens_" + subname,
                            type=model_data["type"],
                            base_url=model_data["base_url"],
                            weights_resource=submodel[subname]["resource"],
                            weights_sha256hash=submodel[subname]["sha256hash"],
                            config_resource=model_data["config"]["resource"],
                            config_sha256hash=model_data["config"]["sha256hash"],
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
                    )
                # check types of models
                if len({model.type for model in models}) > 1:
                    raise ValueError(
                        "All models in an ensemble must be of the same type"
                    )
                # check the mtenn_versions are compatible
                
                
                # set last updated to the oldest of the submodels
                last_updated = min([model.last_updated for model in models])

                ens = EnsembleMLModelSpec(
                    models=models,
                    name=model,
                    type=model_data["type"],
                    last_updated=last_updated,
                    targets=set(model_data["targets"]),
                    mtenn_lower_pin=model_data["mtenn_lower_pin"],
                    mtenn_upper_pin=(
                        model_data["mtenn_upper_pin"]
                        if "mtenn_upper_pin" in model_data
                        else None
                    ),
                )

            except Exception as e:
                warnings.warn(f"Failed to load model {model}, skipping. Error: {e}")
                continue

            ensemble_models[model] = ens

        return ensemble_models


class LocalMLModelSpecBase(MLModelBase):
    """Base class for local model specs"""

    ensemble = False


class LocalMLModelSpec(LocalMLModelSpecBase):
    """
    Model spec for a model instantiated locally, containing file paths to model files
    """

    weights_file: Path = Field(..., description="Weights file path")
    config_file: Optional[Path] = Field(..., description="Optional config file path")
    local_dir: Optional[Path] = Field(
        None,
        description="Local directory for model files, otherwise defaults to pooch.os_cache",
    )


class LocalEnsembleMLModelSpec(LocalMLModelSpecBase):
    """
    Model spec for an ensemble model instantiated locally, containing file paths to model files
    """

    ensemble = True
    models: list[LocalMLModelSpec] = Field(
        ..., description="List of local model specs for ensemble models"
    )

    @property
    def ensemble_size(self):
        return len(self.models)


class MLModelRegistry(BaseModel):
    """
    Model registry for ML models stored remotely, read from a yaml spec file most of the time

    """

    models: dict[str, MLModelSpecBase] = Field(
        ..., description="Models in the model registry, keyed by name"
    )

    def get_models_for_target_and_type(
        self, target: TargetTags, type: ModelType
    ) -> list[MLModelSpecBase]:
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
        List[MLModelSpecBase]
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

    def get_targets_with_models(self) -> list[TargetTags]:
        """
        Get all targets with models

        Returns
        -------
        List[TargetTags]
            List of targets with models
        """
        return list(
            {target.value for model in self.models.values() for target in model.targets}
        )

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
            try:
                model_data = spec[model]
                has_config = "config" in model_data
                is_ensemble = (
                    "remote_ensemble" in model_data and model_data["remote_ensemble"]
                )
                if is_ensemble:
                    ens_models = RemoteEnsembleHelper(
                        manifest_url=model_data["manifest_url"]
                    ).to_ensemble_spec()
                    models.update(ens_models)

                else:
                    # is a single model
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
            except Exception as e:
                warnings.warn(f"Failed to load model {model}, skipping. Error: {e}")

        return cls(models=models)


# default model registry for all ASAP models
ASAPMLModelRegistry = MLModelRegistry.from_yaml(asap_models_yaml)
