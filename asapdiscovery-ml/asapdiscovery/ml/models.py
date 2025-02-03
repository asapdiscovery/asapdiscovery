import os
import warnings
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union  # noqa: F401
from urllib.parse import urljoin

import mtenn
import pooch
import requests
import yaml
from asapdiscovery.data.services.postera.manifold_data_validation import TargetTags
from asapdiscovery.ml.pretrained_models import asap_models_yaml
from mtenn.config import ModelType
from pydantic.v1 import BaseModel, Field, HttpUrl, validator
from semver import Version


class MLModelBase(BaseModel):
    """
    Base model class for ML models
    """

    class Config:

        # Add custom encoders for semver Versions
        json_encoders = {Version: lambda v: str(v)}

        # Allow arbitrary types so that pydantic will accept Versions
        arbitrary_types_allowed = True

    name: str = Field(..., description="Model name")
    endpoint: Any = Field(
        ..., description="Endpoint for model"
    )  # FIXME: should be Optional[str] but this causes issues with pydantic
    type: ModelType = Field(..., description="Model type")
    last_updated: date = Field(..., description="Last updated datetime")
    targets: Any = Field(
        ..., description="Biological targets of the model"
    )  # FIXME: should be Optional[Set[TargetTags]] but this causes issues with pydantic
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
            endpoint=self.endpoint,
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
        if not check_mtenn_version_set_compatible(models):
            raise ValueError(
                "All models in an ensemble must be compatible with the same mtenn version and current mtenn version"
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

    def pull_plot(
        self, plotname: str, filename: Optional[str] = None, return_as="memory"
    ) -> str:
        """
        Pull plot of model performance from a URL

        Parameters
        ----------
        plotname : str
            Name of plot
        filename : Optional[str], optional
            Filename to save plot to, by default None
        return_as : str, optional
            How to return the plot, either 'memory', 'file' or 'url', by default 'memory'

        Returns
        -------
        str
            Plot data, filename or url
        """
        # check all the base urls are the same
        base_url = self.models[0].base_url
        if not all([model.base_url == base_url for model in self.models]):
            raise ValueError("All models in an ensemble must have the same base url")
        # get plot at baseurl/plotname
        plot_url = urljoin(base_url, plotname)

        # pull using requests to in memory
        try:
            response = requests.get(plot_url)
            response.raise_for_status()
        except Exception as e:
            warnings.warn(
                f"Failed to download plot from {plot_url}, skipping. Error: {e}"
            )
            return None
        # return as memory or file
        if return_as == "memory":
            return response.content
        elif return_as == "file":
            # save to file
            if not filename:
                filename = plotname
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        elif return_as == "url":
            return plot_url
        else:
            raise ValueError("return_as must be 'memory' or 'file' or 'url'")


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


def check_mtenn_version_set_compatible(models: list[MLModelSpecBase]) -> bool:
    """
    Check that all models in the list are compatible with the same mtenn version
    Bit hacky but should work for now.
    """

    try:
        cur_version = Version.parse(mtenn.__version__)
    except AttributeError:
        warnings.warn(
            "No mtenn version found. Assuming compatibility, but note "
            "that this may be incorrect."
        )
        return True

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            # ugly sorry
            model1, model2 = models[i], models[j]
            low_pin1 = (
                model1.mtenn_lower_pin
                if model1.mtenn_lower_pin
                else Version.parse("0.0.0")
            )
            upper_pin1 = (
                model1.mtenn_upper_pin
                if model1.mtenn_upper_pin
                else Version.parse("999.999.999")
            )
            low_pin2 = (
                model2.mtenn_lower_pin
                if model2.mtenn_lower_pin
                else Version.parse("0.0.0")
            )
            upper_pin2 = (
                model2.mtenn_upper_pin
                if model2.mtenn_upper_pin
                else Version.parse("999.999.999")
            )

            # check if versions are compatible with current mtenn version
            m1_version_compat = low_pin1 <= cur_version < upper_pin1
            m2_version_compat = low_pin2 <= cur_version < upper_pin2

            if not m1_version_compat or not m2_version_compat:
                return False

            # now check m1 and m2 are cross compatible
            if not (low_pin1 <= upper_pin2 and low_pin2 <= upper_pin1):
                return False

    return True


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
    last_updated: 2024-01-01

    """

    manifest_url: HttpUrl = Field(..., description="Remote ensemble model url")

    def to_ensemble_spec(self) -> dict[str, EnsembleMLModelSpec]:
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
                    if "targets" in model_data:
                        tar = model_data["targets"]
                        # check not a list of None
                        if not all(tar):
                            targets = None
                        else:
                            targets = set(tar)
                    else:
                        targets = None
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
                            targets=targets,
                            endpoint=model_data["endpoint"],
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
                if not check_mtenn_version_set_compatible(models):
                    raise ValueError(
                        "All models in an ensemble must be compatible with the same mtenn version"
                    )

                # set last updated to the oldest of the submodels
                last_updated = min([model.last_updated for model in models])

                ens = EnsembleMLModelSpec(
                    models=models,
                    name=model,
                    type=model_data["type"],
                    last_updated=last_updated,
                    targets=set(model_data["targets"]),
                    endpoint=model_data["endpoint"],
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
    source_yaml: Optional[str] = Field(
        None, description="Source yaml file for model registry"
    )
    time_updated: datetime = Field(datetime.utcnow(), description="Time last updated")

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
            {target for model in self.models.values() for target in model.targets}
        )

    def get_latest_model_for_target_and_type(
        self,
        target: TargetTags,
        type: ModelType,
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

    def get_latest_model_for_target_and_endpoint(
        self, target: TargetTags, endpoint: str
    ) -> MLModelSpec:
        """
        Get latest model spec for a target and endpoint

        Parameters
        ----------
        target : TargetTags
            Target to get model for
        endpoint : str
            Endpoint to get model for

        Returns
        -------
        MLModelSpec
            Latest model spec
        """
        models = [
            model
            for model in self.models.values()
            if target in model.targets and model.endpoint == endpoint
        ]
        if len(models) == 0:
            warnings.warn(
                f"No models available for target {target} and endpoint {endpoint}"
            )
            return None
        else:
            return max(models, key=lambda model: model.last_updated)

    def get_models_for_endpoint(self, endpoint: str) -> list[MLModelSpec]:
        """
        Get models for a given endpoint

        Parameters
        ----------
        endpoint : str
            Endpoint to get models for

        Returns
        -------
        List[MLModelSpec]
            List of model specs
        """
        return [model for model in self.models.values() if model.endpoint == endpoint]

    def get_latest_model_for_endpoint(self, endpoint: str) -> MLModelSpec:
        """
        Get latest model spec for a given endpoint

        Parameters
        ----------
        endpoint : str
            Endpoint to get model for

        Returns
        -------
        MLModelSpec
            Latest model spec
        """
        models = self.get_models_for_endpoint(endpoint)
        if len(models) == 0:
            warnings.warn(f"No models available for endpoint {endpoint}")
            return None
        else:
            return max(models, key=lambda model: model.last_updated)

    def get_models_without_target(self) -> list[MLModelSpec]:
        """
        Get models without a target

        Returns
        -------
        List[MLModelSpec]
            List of model specs
        """
        return [model for model in self.models.values() if not any(model.targets)]

    def get_endpoints(self) -> list[str]:
        """
        Get list of endpoints

        Returns
        -------
        List[str]
            List of endpoints
        """
        return list({model.endpoint for model in self.models.values()})

    def get_endpoint_target_mapping(self) -> dict[str, list[TargetTags]]:
        """
        Get mapping of endpoints to targets

        Returns
        -------
        dict[str, list[TargetTags]]
            Mapping of endpoints to targets
        """
        map = defaultdict(list)
        for model in self.models.values():
            map[model.endpoint].extend(list(model.targets))

        # uniquify
        new_map = {}
        for k, v in map.items():
            if any(v):
                new_map[k] = list(set(v))
            else:
                new_map[k] = None

        return new_map

    def get_endpoints_for_target(
        self, target: TargetTags, include_generic=True
    ) -> list[str]:
        """
        Get list of endpoints for a target

        Parameters
        ----------
        target : TargetTags
            Target to get endpoints for
        include_generic : bool, optional
            Whether to include generic endpoints, by default True

        Returns
        -------
        List[str]
            List of endpoints
        """
        endpts = list(
            {
                model.endpoint
                for model in self.models.values()
                if target in model.targets
            }
        )
        if include_generic:
            extra = self.get_endpoints_for_target(None, include_generic=False)
            endpts.extend(extra)
        # uniquify
        endpts = list(set(endpts))
        return endpts

    def endpoint_has_target(self, endpoint: str) -> bool:
        """
        Check if an endpoint has a target

        Parameters
        ----------
        endpoint : str
            Endpoint to check
        target : TargetTags
            Target to check

        Returns
        -------
        bool
            Whether the endpoint has the target
        """
        return self.get_endpoint_target_mapping().get(endpoint) is not None

    def get_target_endpoint_mapping(self) -> dict[TargetTags, list[str]]:
        """
        Get mapping of targets to endpoints

        Returns
        -------
        dict[TargetTags, list[str]]
            Mapping of targets to endpoints
        """
        map = defaultdict(list)
        for model in self.models.values():
            for target in model.targets:
                map[target].append(model.endpoint)

        # uniquify
        new_map = {}
        for k, v in map.items():
            if any(v):
                new_map[k] = list(set(v))
            else:
                new_map[k] = None

        return new_map

    def get_latest_model_for_target_type_and_endpoint(
        self, target: str, type: str, endpoint: str
    ) -> MLModelSpec:
        """
        Get latest model spec for a target, type and endpoint

        Parameters
        ----------
        target : TargetTags
            Target to get model for
        type : ModelType
            Type of model to get
        endpoint : str
            Endpoint to get model for

        Returns
        -------
        MLModelSpec
            Latest model spec
        """
        models = [
            model
            for model in self.models.values()
            if target in model.targets
            and model.endpoint == endpoint
            and model.type == type
        ]
        if len(models) == 0:
            warnings.warn(
                f"No models available for target {target}, endpoint {endpoint} and type {type}"
            )
            return None
        else:
            return max(models, key=lambda model: model.last_updated)

    def get_model_types_for_endpoint(self, endpoint: str) -> list[str]:
        """
        Get model types for an endpoint

        Parameters
        ----------
        endpoint : str
            Endpoint to get model types for

        Returns
        -------
        List[str]
            List of model types
        """
        return list(
            {model.type for model in self.models.values() if model.endpoint == endpoint}
        )

    def reccomend_models_for_target(self, target: TargetTags) -> list[MLModelSpec]:
        """
        Get reccomended models for a target, including generic models without a target
        """
        if target not in TargetTags.get_values():
            raise ValueError(
                f"Target {target} not valid, must be one of {TargetTags.get_values()}"
            )
        epts = self.get_endpoints_for_target(target, include_generic=True)
        models = []
        for p in epts:
            for m in self.get_model_types_for_endpoint(p):
                if ASAPMLModelRegistry.endpoint_has_target(p):
                    mod = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(
                        target, m, p
                    )
                else:
                    mod = ASAPMLModelRegistry.get_latest_model_for_target_type_and_endpoint(
                        None, m, p
                    )
                models.append(mod)

        # clean for None
        models = [m for m in models if m is not None]
        return models

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

    def update_registry(self):
        """
        Refresh the model registry by checking for new models
        """
        if not self.source_yaml:
            raise ValueError(
                "No source yaml file provided for model registry, cannot update"
            )
        new_models = self.parse_yaml_to_models_dict(self.source_yaml)
        self.models = new_models
        self.time_updated = datetime.utcnow()

    @staticmethod
    def parse_yaml_to_models_dict(
        yaml_file: Union[str, Path]
    ) -> dict[str, MLModelSpecBase]:
        """
        Parse models registry from yaml spec file

        Parameters
        ----------
        yaml_file : Union[str, Path]

        Returns
        -------
        dict[str, MLModelSpecBase]
            Dictionary of model specs
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
                        endpoint=model_data["endpoint"],
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

        return models

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "MLModelRegistry":
        """
        Make model registry from yaml spec file
        """
        models = cls.parse_yaml_to_models_dict(yaml_file)
        return cls(models=models, source_yaml=yaml_file)


_asap_ml_debug = True if os.getenv("ASAP_ML_DEBUG") else False

if _asap_ml_debug:
    print("ASAP_ML_DEBUG MODE ENABLED")
    print(f"ASAP Models YAML: {asap_models_yaml}")

    # load raw, will show a bunch of warnings if things not working
    # default model registry for all ASAP models
    ASAPMLModelRegistry = MLModelRegistry.from_yaml(asap_models_yaml)
    print("ASAP ML Model Registry loaded")
else:
    # load with UserWarning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ASAPMLModelRegistry = MLModelRegistry.from_yaml(asap_models_yaml)
