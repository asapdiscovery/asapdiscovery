from typing import Optional

from asapdiscovery.data.schema.ligand import Ligand, LigandIdentifiers
from asapdiscovery.data.services.postera.molecule_set import MoleculeSetAPI
from asapdiscovery.data.services.services_config import PosteraSettings
from pydantic import BaseModel, Field


class PosteraFactory(BaseModel):
    settings: PosteraSettings = Field(default_factory=PosteraSettings)
    molecule_set_name: Optional[str] = Field(
        "", description="Name of the molecule set to pull from Postera"
    )

    def pull(self) -> list[Ligand]:
        """
        Pull molecules from a Postera molecule set

        Returns
        -------
        List[Ligand]
            List of ligands
        """
        ms_api = MoleculeSetAPI.from_settings(self.settings)
        mols, _ = ms_api.get_molecules_from_id_or_name(self.molecule_set_name)

        # check if there are any custom columns in this moleculeset
        standard_columns = ["smiles", "id", "idx", "label"]
        custom_data_columns = [
            col for col in mols.columns if not col in standard_columns
        ]

        ligands = []
        for _, mol in mols.iterrows():
            # create the ligand with relevant metadata
            ligand = Ligand.from_smiles(
                compound_name=mol.id,
                smiles=mol.smiles,
                ids=LigandIdentifiers(manifold_api_id=mol.id),
            )

            # now append custom data to the Ligand's tags, if there is any
            tags = {}
            for custom_col in custom_data_columns:
                if mol[custom_col] is None:
                    mol[custom_col] = ""
                tags[custom_col] = mol[custom_col]

            ligand.tags = tags
            ligands.append(ligand)
        return ligands

    def pull_all(self) -> list[Ligand]:
        """
        Pull all molecules from all Postera molecule sets

        Returns
        -------
        List[Dict]
            List of dictionaries, where each dict is a moleculeset with metadata and ligand data
        """
        from rich.progress import track

        ms_api = MoleculeSetAPI.from_settings(self.settings)
        available_msets = ms_api.list_available()

        all_mset_data = []
        for mset_uuid, _ in track(
            available_msets.items(),
            total=len(available_msets),
            description=f"Processing {len(available_msets)} available moleculesets..",
        ):
            # gather metadata of this mset
            mset_metadata = ms_api.get(
                mset_uuid
            )  # need to use UUID here instead of name for postera API reasons.

            # gather compound data contained in this mset
            self.molecule_set_name = mset_uuid
            mset_compound_data = self.pull()

            # add to metadata, and add the whole thing to the data bucket
            mset_metadata["ligand_data"] = mset_compound_data
            all_mset_data.append(mset_metadata)

        return all_mset_data
