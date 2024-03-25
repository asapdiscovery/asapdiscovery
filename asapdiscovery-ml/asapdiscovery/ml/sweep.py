from pathlib import Path

from asapdiscovery.ml.trainer import Trainer
from pydantic import Field, validator
import wandb
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

    num_sweeps: int = Field(1, description="Number of different sweep configs to try.")

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

    def wandb_init(self):
        """
        Overload the Trainer.wandb_init function to check if a sweep exists, and start
        one if not.
        """

        # If sweep_id_fn exists, load sweep_id from there
        sweep_id_fn = self.output_dir / "sweep_id"
        if sweep_id_fn.exists():
            sweep_id = sweep_id_fn.read_text().strip()
        else:
            sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.wandb_project)
            sweep_id_fn.write_text(sweep_id)

        # Start W&B agent
        wandb.agent(
            sweep_id, function=..., project=self.wandb_project, count=self.num_sweeps
        )
        # function here needs to be something I think, but not sure how to make that work
