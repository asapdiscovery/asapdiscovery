import warnings
from typing import Optional

from asapdiscovery.data.schema.ligand import Ligand, LigandIdentifiers
from asapdiscovery.data.services.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.services.services_config import PosteraSettings
from pydantic.v1 import BaseModel, Field


class PosteraFactory(BaseModel):
    settings: PosteraSettings = Field(default_factory=PosteraSettings)
    molecule_set_name: Optional[str] = Field(
        None, description="Name of the molecule set to pull from Postera"
    )
    molecule_set_id: Optional[str] = Field(
        None, description="ID of the molecule set to pull from Postera"
    )

    @staticmethod
    def _pull_molecule_set(
        ms_api: MoleculeSetAPI,
        molecule_set_id: Optional[str] = None,
        molecule_set_name: Optional[str] = None,
    ) -> list[Ligand]:
        if molecule_set_id is None and molecule_set_name is None:
            raise ValueError("You must provide either a molecule set name or ID")

        mols, _ = ms_api.get_molecules_from_id_or_name(
            name=molecule_set_name, id=molecule_set_id
        )

        # check if there are any custom columns in this moleculeset
        standard_columns = ["smiles", "id", "idx", "label"]
        custom_data_columns = [
            col for col in mols.columns if col not in standard_columns
        ]

        ligands = []
        for _, mol in mols.iterrows():
            # create the ligand with relevant metadata
            try:
                smiles = mol.smiles
                ligand = Ligand.from_smiles(
                    compound_name=mol.id,
                    smiles=smiles,
                    ids=LigandIdentifiers(manifold_api_id=mol.id),
                )

                # now append custom data to the Ligand's tags, if there is any
                tags = {}
                for custom_col in custom_data_columns:
                    if custom_col in Ligand.__fields__.keys():
                        warnings.warn(
                            f"Custom column name {custom_col} is already a field in Ligand, skipping.."
                        )
                        continue

                    if mol[custom_col] is None:
                        mol[custom_col] = ""

                    tags[custom_col] = mol[custom_col]

                ligand.tags = tags
                ligands.append(ligand)
            except Exception as e:  # noqa: E722
                warnings.warn(
                    f"Failed to create ligand from smiles: {smiles}, error is: {e}"
                )
        return ligands

    def pull(self) -> list[Ligand]:
        """
        Pull molecules from a Postera molecule set

        Returns
        -------
        List[Ligand]
            List of ligands
        """
        ms_api = MoleculeSetAPI.from_settings(self.settings)
        return self._pull_molecule_set(
            ms_api, self.molecule_set_id, self.molecule_set_name
        )

    def pull_all(self, progress=True) -> list[dict]:
        """
        Pull all molecules from all Postera molecule sets

        Parameters
        ----------
        progress: bool, optional
            Whether to show a progress bar, by default True

        Returns
        -------
        List[Dict]
            List of dictionaries, where each dict is a moleculeset with metadata and ligand data
        """
        from rich.progress import track

        ms_api = MoleculeSetAPI.from_settings(self.settings)
        available_msets = ms_api.list_available()

        if progress:
            wrapper = track
        else:
            wrapper = lambda x, **kwargs: x  # noqa: E731

        all_mset_data = []
        for mset_uuid, _ in wrapper(
            available_msets.items(),
            total=len(available_msets),
            description=f"Processing {len(available_msets)} available moleculesets..",
        ):
            # gather metadata of this mset
            mset_metadata = ms_api.get(
                mset_uuid
            )  # need to use UUID here instead of name for postera API reasons.

            # gather compound data contained in this mset
            mset_compound_data = self._pull_molecule_set(
                ms_api, molecule_set_id=mset_uuid
            )

            # add to metadata, and add the whole thing to the data bucket
            mset_metadata["ligand_data"] = mset_compound_data
            all_mset_data.append(mset_metadata)

        return all_mset_data
