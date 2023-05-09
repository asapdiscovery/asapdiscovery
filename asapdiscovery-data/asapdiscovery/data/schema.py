from datetime import date
from io import StringIO
from pathlib import Path
from pydantic import BaseModel, ValidationError, validator, Field
from typing import Optional, Union
from .validation import is_valid_smiles

from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    oechem,
    save_openeye_pdb_string,
)
from asapdiscovery.docking.modeling import du_to_complex


# From FAH ###################################
class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ExperimentalCompoundData(Model):
    compound_id: str = Field(
        None,
        description="The unique compound identifier (PostEra or enumerated ID)",
    )

    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    racemic: bool = Field(
        False,
        description="If True, this experiment was performed on a racemate; if False, the compound was enantiopure.",
    )

    achiral: bool = Field(
        False,
        description="If True, this compound has no chiral centers or bonds, by definition enantiopure",
    )

    absolute_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure and stereochemistry recorded in SMILES is correct",
    )

    relative_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure, but unknown if stereochemistry recorded in SMILES is correct",
    )

    date_created: date = Field(
        None, description="Date the molecule was created."
    )

    experimental_data: dict[str, float] = Field(
        dict(),
        description='Experimental data fields, including "pIC50" and uncertainty (either "pIC50_stderr" or  "pIC50_{lower|upper}"',
    )


class ExperimentalCompoundDataUpdate(Model):
    """A bundle of experimental data for compounds (racemic or enantiopure)."""

    compounds: list[ExperimentalCompoundData]


########################################


class CrystalCompoundData(BaseModel):
    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    compound_id: str = Field(
        None, description="The unique compound identifier of the ligand."
    )

    dataset: str = Field(
        None, description="Dataset name from Fragalysis (name of structure)."
    )

    str_fn: str = Field(None, description="Filename of the PDB structure.")

    sdf_fn: str = Field(None, description="Filename of the SDF file")
    active_site_chain: str = Field(
        None, description="Chain identifying the active site of interest."
    )
    output_name: str = Field(None, description="Name of output structure.")
    active_site: str = Field(
        None, description="OpenEye formatted active site residue."
    )
    oligomeric_state: str = Field(
        None, description="Oligomeric state of the asymmetric unit."
    )
    chains: list = Field(
        None, description="List of chainids in the asymmetric unit."
    )
    protein_chains: list = Field(
        None, description="List of chains corresponding to protein residues."
    )

    series: str = Field(
        None,
        description="Name of COVID Moonshot series associated with this molecule",
    )


class PDBStructure(Model):
    pdb_id: str = Field(None, description="PDB identification code.")
    str_fn: str = Field(None, description="Filename of local PDB structure.")


class EnantiomerPair(Model):
    active: ExperimentalCompoundData = Field(description="Active enantiomer.")
    inactive: ExperimentalCompoundData = Field(
        description="Inactive enantiomer."
    )


class EnantiomerPairList(Model):
    pairs: list[EnantiomerPair]


class ProvenanceBase(Model):
    ...


#########################################


class Ligand(BaseModel):
    smiles: str
    id: str = Field(None, description="the compound identifier")
    vc_id_postera: str = Field(
        None, description="the PostERA master compound ID"
    )
    moonshot_compound_id: str = Field(
        None, description="the Moonshot compound ID"
    )
    target_id: str = Field(None, description="the target protein ID")
    source: str = None

    ligand_provenance: Optional[ProvenanceBase] = None

    @validator("smiles")
    def smiles_must_be_valid(cls, v):
        if not is_valid_smiles(v):
            raise ValueError("Invalid SMILES string")

    def to_sdf():
        ...

    def to_smiles():
        ...

    def to_pdb():
        ...

    def to_design_unit():
        ...

    def to_oemol():
        ...

    @staticmethod
    def from_smiles():
        ...

    @staticmethod
    def from_multiligand_sdf():
        ...

    @staticmethod
    def from_sdf(sdf_fn: Union[str, Path]):
        if not is_single_molecule_sdf(sdf_fn):
            raise ValueError("SDF file must contain a single molecule")
        mol = load_openeye_molecule(sdf_fn)
        smiles = get_canonical_smiles(mol)
        return Ligand(smiles=smiles, source=sdf_fn)


################################################################################
class Target(BaseModel):
    id: str = Field(description="ID for the target.")
    pdb_code: Optional[str] = Field("", description="RCSB PDB code.")
    chain: str = Field(description="Which chain the protein is in.")
    source: str = Field(description="Full PDB file to construct target.")
    reference_ligand: Optional[Ligand] = Field(
        None, description="Ligand object bound to target."
    )
    protein_provenance: Optional[ProvenanceBase] = Field(
        None, description="Provenance."
    )

    def get_ligand(self):
        if not self.reference_ligand:
            raise ValueError("Target does not have a reference ligand")
        return self.reference_ligand

    @staticmethod
    def from_pdb(pdb_fn, id):
        """
        Construct a Target from a PDB file.

        Parameters
        ----------
        pdb_fn : Union[str, Path]
            PDB file to use for construction
        id : str
            Target ID

        Returns
        -------
        Target
        """
        # Get PDB source
        source = Path(pdb_fn).open().read()

        # Load OE mol
        pdb_mol = load_openeye_pdb(pdb_fn)

        # Don't just use from_oemol because we already have the actual PDB source
        # Take first chain ID in the PDB file
        # (is this what we actually want to do?)
        chain = next(oechem.OEHierView(pdb_mol).GetChains()).GetChainID()

        # Construct ligand using already loaded OEMol
        reference_ligand = Ligand.from_oemol(pdb_mol)

        return Target(
            id=id, chain=chain, source=source, reference_ligand=reference_ligand
        )

    @staticmethod
    def from_oedu(du_fn, id):
        """
        Construct a Target from an oedu file.

        Parameters
        ----------
        du_fn : Union[str, Path]
            oedu file to use for construction
        id : str
            Target ID

        Returns
        -------
        Target
        """
        # Load DU
        du = oechem.OEDesignUnit()
        if not oechem.OEReadDesignUnit(str(du_fn), du):
            raise RuntimeError(f"Unable to read OEDU file {du_fn}")

        return Target.from_design_unit(du, id)

    @staticmethod
    def from_design_unit(du, id):
        """
        Construct a Target from a DesignUnit object.

        Parameters
        ----------
        du : oechem.OEDesignUnit
            DesignUnit input
        id : str
            Target ID

        Returns
        -------
        Target
        """
        # Convert DU to OEMol
        pdb_mol = du_to_complex(du, include_solvent=True)

        # Use from_oemol function
        return Target.from_oemol(pdb_mol, id)

    @staticmethod
    def from_oemol(pdb_mol, id):
        """
        Construct a Target from an OEMol object.

        Parameters
        ----------
        mol : oechem.OEMol
            OEMol input
        id : str
            Target ID

        Returns
        -------
        Target
        """
        # Take first chain ID in the PDB file
        # (is this what we actually want to do?)
        chain = next(oechem.OEHierView(pdb_mol).GetChains()).GetChainID()

        # Get PDB string
        source = save_openeye_pdb_string(pdb_mol)

        # Construct ligand using already loaded OEMol
        reference_ligand = Ligand.from_oemol(pdb_mol)

        return Target(
            id=id, chain=chain, source=source, reference_ligand=reference_ligand
        )
