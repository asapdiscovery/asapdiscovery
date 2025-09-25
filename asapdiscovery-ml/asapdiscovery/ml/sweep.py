from functools import partial
from pathlib import Path

import wandb
import yaml
from asapdiscovery.ml.trainer import Trainer
from pydantic.v1 import Field, validator


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

    force_new_sweep: bool = Field(
        False, description="Start a new sweep even if an existing sweep_id is present."
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

    def start_continue_sweep(self, start_only=False):
        """
        Check for existing sweep, and start one if not. This is the function to run
        when doing a sweep.

        Parameters
        ----------
        start_only : bool, default=False
            Don't start any sweep runs, just initialize
        """

        # If sweep_id_fn exists, load sweep_id from there
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sweep_id_fn = self.output_dir / "sweep_id"
        if sweep_id_fn.exists() and (not self.force_new_sweep):
            sweep_id = sweep_id_fn.read_text().strip()
        else:
            sweep_id = wandb.sweep(sweep=self.sweep_config, project=self.wandb_project)
            sweep_id_fn.write_text(sweep_id)

        if start_only:
            return

        # Set up partial function so we can pass this Sweeper object along to the
        #  dispatch function
        sweep_func = partial(Sweeper._sweep_dispatch, self)

        # Start W&B agent
        wandb.agent(
            sweep_id,
            function=sweep_func,
            project=self.wandb_project,
            count=1,
        )

    def _update_from_wandb_config(self):
        """
        Parse the W&B sweep config and update the internal config objects appropriately.
        """

        # Clone self into a dict so we can update stuff
        new_trainer_dict = self.dict()

        # Decompose parameter names into nested dict
        config_update_dict = {}
        for k, v in wandb.config.items():
            d = config_update_dict
            while "." in k:
                accession, _, k = k.partition(".")
                if accession not in d:
                    d[accession] = {}
                d = d[accession]

            d[k] = v

        # Loop through keys and try to update configs
        failed_configs = []
        for config_name, config_d in config_update_dict.items():
            if config_name not in new_trainer_dict:
                failed_configs.append(config_name)
                continue

            try:
                new_trainer_dict[config_name].update(config_d)
            except AttributeError:
                new_trainer_dict[config_name] = config_d

        if len(failed_configs) > 0:
            raise AttributeError(
                f"Could not assign values for these keys: {failed_configs}"
            )

        return Sweeper(**new_trainer_dict)

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
        sweeper = sweeper._update_from_wandb_config()

        # Update W&B config to include everything from all the Trainer configs
        # Don't serialize input_data for confidentiality/size reasons
        ds_config = sweeper.ds_config.dict()
        del ds_config["input_data"]
        config = sweeper.dict()
        config["ds_config"] = ds_config
        wandb.config.update(config)

        # Get Trainer config dict (before initialization so we don't have extra stuff)
        trainer_config_dict = sweeper.dict()
        # Get rid of Sweeper-specific args
        del trainer_config_dict["sweep_config"]
        del trainer_config_dict["force_new_sweep"]

        # Temporarily un-set use_wandb flag to avoid confusing the initialize method
        sweeper.use_wandb = False
        # Run initialize to build all the objects
        sweeper.initialize()
        sweeper.use_wandb = True

        # Update output directory to save in sweep run id dir
        trainer_config_dict["output_dir"] = sweeper.output_dir
        t = Trainer(**trainer_config_dict)
        (sweeper.output_dir / "trainer.json").write_text(t.json())

        # Log dataset splits
        for split, table in zip(
            ["train", "val", "test"], sweeper._make_wandb_ds_tables()
        ):
            wandb.log({f"dataset_splits/{split}": table})

        # Finally run training
        sweeper.train()
