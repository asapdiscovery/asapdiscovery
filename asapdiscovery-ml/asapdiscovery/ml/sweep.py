from pathlib import Path

from asapdiscovery.ml.trainer import Trainer
from pydantic import Field, validator
import yaml


class Sweeper(Trainer):
    """
    Subclass of Trainer that handles running a W&B sweep. Most of the functionality is
    be the same, but this class also handles sweep starting as necessary.
    """

    # Required parameters for setting up the sweep
    sweep_config: dict = Field(
        ...,
        description=(
            "W&B sweep configuration. See the W&B docs for more information "
            "on what this should look like."
        ),
    )

    @validator("sweep_config", pre=True)
    def load_config(cls, v):
        """
        Support for loading sweep config YAML files.
        """

        # Already in dict form
        if isinstance(v, dict):
            return v

        # Load from file
        return yaml.safe_load(Path(v).read_text())
