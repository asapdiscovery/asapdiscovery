from typing import TYPE_CHECKING

import click
import pandas as pd
import rich

if TYPE_CHECKING:
    from asapdiscovery.data.schema.ligand import Ligand


def print_header(console: "rich.Console"):
    """Print an ASAP-Alchemy header"""

    console.line()
    console.rule("ASAP-Alchemy")
    console.line()


def pull_from_postera(molecule_set_name: str):
    """
    A convenience method with tucked imports to avoid importing Postera tools when not needed.

    Args:
        The name of the molecule set which should be pulled from postera

    Returns:
        A list of Ligands extracted from postera molecule set.
    """
    from asapdiscovery.data.services.postera.postera_factory import PosteraFactory
    from asapdiscovery.data.services.services_config import PosteraSettings

    # this will pull the settings from environment variables
    settings = PosteraSettings()
    return PosteraFactory(settings=settings, molecule_set_name=molecule_set_name).pull()


def upload_to_postera(
    molecule_set_name: str, target: str, absolute_dg_predictions: pd.DataFrame
):
    """
    A convenience method to format predicted absolute DG values using Alchemy and upload to postera with tucked imports
    to avoid importing Postera tools.

    Args:
        molecule_set_name: The name of the molecule set in postera the results should be attached to.
        target: The name of the biological target this result is associated with.
        absolute_dg_predictions: The dataset of absolute dg predictions created by asap-alchemy.
    """
    from enum import Enum

    from asapdiscovery.alchemy.predict import dg_to_postera_dataframe
    from asapdiscovery.data.services.postera.manifold_data_validation import (
        rename_output_columns_for_manifold,
    )
    from asapdiscovery.data.services.postera.postera_uploader import PosteraUploader
    from asapdiscovery.data.services.services_config import PosteraSettings

    # mock an enum to specify which columns are allowed
    class AlchemyResults(str, Enum):
        SMILES = "SMILES"
        LIGAND_ID = "Ligand_ID"
        COMPUTED_BIOCHEMICAL_ACTIVITY_FEC = "computed-FEC-pIC50"
        COMPUTED_BIOCHEMICAL_ACTIVITY_FEC_UNCERTAINTY = "computed-FEC-uncertainty-pIC50"

    # convert the dg values to pIC50 with the expected names
    postera_df = dg_to_postera_dataframe(absolute_predictions=absolute_dg_predictions)
    result_df = rename_output_columns_for_manifold(
        df=postera_df,
        target=target,
        output_enums=[AlchemyResults],
        manifold_validate=True,
        drop_non_output=True,
        allow=[
            AlchemyResults.SMILES.value,
            AlchemyResults.LIGAND_ID.value,
        ],
    )

    postera_uploader = PosteraUploader(
        settings=PosteraSettings(),
        molecule_set_name=molecule_set_name,
        id_field=AlchemyResults.LIGAND_ID.value,
        smiles_field=AlchemyResults.SMILES.value,
    )

    _, _, _ = postera_uploader.push(df=result_df)


def get_cdd_molecules(
    protocol_name: str, defined_stereo_only: bool = True, remove_covalent: bool = True
) -> list["Ligand"]:
    """
    Search the CDD protocol for molecules with experimental values and return a list of asapdiscovery ligands.

    Notes:
        The ligands will contain a tag which can be used to identify them as experimental compounds later.

    Args:
        protocol_name: The name of the experimental protocol in CDD we should extract molecules from.
        defined_stereo_only: Only return ligands which have fully defined stereochemistry
        remove_covalent: If `True` remove potential covalent ligands from the protocol based on the presence of warheads
            found via smarts matches.

    Returns:
        A list of molecules with experimental data.
    """
    from asapdiscovery.alchemy.predict import download_cdd_data
    from asapdiscovery.data.schema.ligand import Ligand

    # get all molecules with data for the protocol
    cdd_data = download_cdd_data(protocol_name=protocol_name)

    ref_ligands = []
    for _, row in cdd_data.iterrows():
        asap_mol = Ligand.from_smiles(
            smiles=row["Smiles"],
            compound_name=row["Molecule Name"],
            cxsmiles=row["CXSmiles"],
        )
        asap_mol.tags["cdd_protocol"] = protocol_name
        asap_mol.tags["experimental"] = "True"
        ref_ligands.append(asap_mol)

    if defined_stereo_only:
        # remove ligands with undefined or non-absolute stereochemistry
        defined_ligands = []
        from openff.toolkit import Molecule
        from openff.toolkit.utils.exceptions import UndefinedStereochemistryError
        from rdkit import Chem

        for mol in ref_ligands:
            try:
                # this checks for any undefined stereo centers
                _ = Molecule.from_smiles(mol.smiles)
                # check for non-absolute centers using the enhanced stereo smiles
                rdmol = Chem.MolFromSmiles(mol.tags["cxsmiles"])
                groups = rdmol.GetStereoGroups()
                for stereo_group in groups:
                    if (
                        stereo_group.GetGroupType()
                        != Chem.StereoGroupType.STEREO_ABSOLUTE
                    ):
                        raise UndefinedStereochemistryError("missing absolute stereo")
                # if we make it through all checks add the molecule
                defined_ligands.append(mol)

            except UndefinedStereochemistryError:
                continue

        ref_ligands = defined_ligands

    if remove_covalent:
        # remove any ligands which contain potential covalent warheads
        non_covalent_ligands = []
        for mol in ref_ligands:
            if not has_warhead(ligand=mol):
                non_covalent_ligands.append(mol)

        ref_ligands = non_covalent_ligands

    return ref_ligands


def has_warhead(ligand: "Ligand") -> bool:
    """
    Check if the molecule has a potential covalent warhead based on the presence of some simple SMARTS patterns.

    Args:
        ligand: The ligand which we should check for potential covalent warheads

    Returns:
        `True` if the ligand has a possible warhead else `False`.

    Notes:
        The list of possible warheads is not exhaustive and so the molecule may still be a covalent ligand.
    """
    from rdkit import Chem

    covalent_warhead_smarts = {
        'acrylamide': '[C;H2:1]=[C;H1]C(N)=O',
        'acrylamide_adduct': 'NC(C[C:1]S)=O',
        'chloroacetamide': 'Cl[C;H2:1]C(N)=O',
        'chloroacetamide_adduct': 'S[C:1]C(N)=O',
        'vinylsulfonamide': 'NS(=O)([C;H1]=[C;H2:1])=O',
        'vinylsulfonamide_adduct': 'NS(=O)(C[C:1]S)=O',
        'nitrile': 'N#[C:1]-[*]',
        'nitrile_adduct': 'C-S-[C:1](=N)',
    }
    rdkit_mol = ligand.to_rdkit()
    for smarts in covalent_warhead_smarts.values():
        if rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return True
    return False


class SpecialHelpOrder(click.Group):
    # from https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super().__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super().get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super().list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 1), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop("help_priority", 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator
