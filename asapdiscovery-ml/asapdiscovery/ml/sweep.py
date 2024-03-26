from functools import partial
from pathlib import Path

from asapdiscovery.ml.trainer import Trainer
from pydantic import Field, validator
import wandb
import yaml


class Sweeper(Trainer):
    """
    Subclass of Trainer that handles running a W&B sweep. Most of the functionality is
    the same, but this class also handles sweep starting as necessary.

    Note that the parameter names should match the field names they are supposed to be
    assigned to (eg to sweep across dataset splitting types, you would have
    "ds_splitter_config.split_type": {"values": ["random", "temporal"]} as an entry in
    the "parameters" section of the sweep_config).
    """

    # Required parameters for setting up the sweep
    sweep_config: dict = Field(
        ...,
        description=(
            "W&B sweep configuration. See the W&B docs for more information "
            "on what this should look like. Note that the parameter names should match "
            "the field names they are supposed to be assigned to "
            "(eg ds_splitter_config.split_type)."
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

    def start_continue_sweep(self):
        """
        Check for existing sweep, and start one if not. This is the function to run
        when doing a sweep.
        """

        # If sweep_id_fn exists, load sweep_id from there
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sweep_id_fn = self.output_dir / "sweep_id"
        if sweep_id_fn.exists():
            sweep_id = sweep_id_fn.read_text().strip()
        else:
            sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.wandb_project)
            sweep_id_fn.write_text(sweep_id)

        # Set up partial function so we can pass this Sweeper object along to the
        #  dispatch function
        sweep_func = partial(Sweeper._sweep_dispatch, self)

        # Start W&B agent
        wandb.agent(
            sweep_id,
            function=sweep_func,
            project=self.wandb_project,
            count=self.num_sweeps,
        )

    def _update_from_wandb_config(self):
        """
        Parse the W&B sweep config and update the internal config objects appropriately.
        """

        # Loop through keys and try to assign values
        failed_keys = []
        for k, v in wandb.config.items():
            obj = self
            orig_key = k
            # Can't do nested accessions naturally, so keep taking the leftmost property
            #  and updating the object we're pointing at
            while not hasattr(obj, k):
                accession, _, k = k.partition(".")
                obj = getattr(obj, accession)

            # Reached the bottom, so assign value
            try:
                setattr(obj, k, v)
            except AttributeError:
                failed_keys.append(orig_key)

        if len(failed_keys) > 0:
            raise AttributeError(
                f"Could not assign values for these keys: {failed_keys}"
            )

    @staticmethod
    def _sweep_dispatch(sweeper: "Sweeper"):
        """
        This is a dispatch function to hand to the wandb.agent function call and should
        probably not be run outside of that context. This function will need to handle:

        1. Parsing the config determined by the W&B sweep
        2. Updating the underlying Trainer configs with these parameters
        3. Running training

        Parameters
        ----------
        sweeper: Sweeper
            This is the Sweeper class that will ultimately run the training
        """

        # Get config from sweep agent
        run_id = wandb.init().id

        # Update output_dir to avoid overwriting stuff
        sweeper.output_dir = sweeper.output_dir / run_id

        # Update internal configs from sweep config
        sweeper._update_from_wandb_config()

        # Temporary un-set use_wandb flag to avoid confusing the initialize method
        sweeper.use_wandb = False
        # Run initialize to build all the objects
        sweeper.initialize()
        sweeper.use_wandb = True

        # Finally run training
        sweeper.train()
