from openeye import oechem, oedocking, oespruce

from .datasets.utils import (
    load_openeye_pdb,
    load_openeye_sdf,
    split_openeye_mol,
)


def du_to_complex(du):
    """
    Convert OEDesignUnit to OEGraphMol containing the protein and ligand from
    `du`.

    Parameters
    ----------
    du : oechem.OEDesignUnit
        OEDesignUnit object to extract from.

    Returns
    -------
    oechem.OEGraphMol
        Molecule with protein and ligand from `du`
    """
    complex_mol = oechem.OEGraphMol()
    du.GetComponents(
        complex_mol,
        oechem.OEDesignUnitComponents_Protein
        | oechem.OEDesignUnitComponents_Ligand,
    )

    return complex_mol


def make_du_from_new_lig(
    initial_complex,
    new_lig,
    ref_prot=None,
    split_initial_complex=True,
    split_ref=True,
):
    """
    Create an OEDesignUnit object from the protein component of
    `initial_complex` and the ligand in `new_lig`. Optionally pass in `ref_prot`
    to align the protein part of `initial_complex`.

    Parameters
    ----------
    initial_complex : Union[oechem.OEGraphMol, str]
        Initial complex loaded straight from a PDB file. Can contain ligands,
        waters, cofactors, etc., which will be removed. Can also pass a PDB
        filename instead.
    new_lig : Union[oechem.OEGraphMol, str]
        New ligand molecule (loaded straight from a file). Can also pass a PDB
        or SDF filename instead.
    ref_prot : Union[oechem.OEGraphMol, str], optional
        Reference protein to which the protein part of `initial_complex` will
        be aligned. Can also pass a PDB filename instead.
    split_initial_complex : bool, default=True
        Whether to split out protein from `initial_complex`. Setting this to
        False will save time on protein prep if you've already isolated the
        protein.
    split_ref : bool, default=True
        Whether to split out protein from `ref_prot`. Setting this to
        False will save time on protein prep if you've already isolated the
        protein.

    Returns
    -------
    oechem.OEDesignUnit
    """

    ## Load initial_complex from file if necessary
    if type(initial_complex) is str:
        initial_complex = load_openeye_pdb(initial_complex, alt_loc=True)
        ## If alt locations are present in PDB file, set positions to highest
        ##  occupancy ALT
        alf = oechem.OEAltLocationFactory(initial_complex)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(initial_complex)

    ## Load ligand from file if necessary
    if type(new_lig) is str:
        if new_lig[-3:] == "sdf":
            parse_func = load_openeye_sdf
        elif new_lig[-3:] == "pdb":
            parse_func = load_openeye_pdb
        else:
            raise ValueError(f"Unknown file format: {new_lig}")

        new_lig = parse_func(new_lig)
    ## Update ligand dimensions in case its internal dimensions property is set
    ##  as 2 for some reason (eg created from SMILES)
    new_lig.SetDimension(3)

    ## Load reference protein from file if necessary
    if type(ref_prot) is str:
        ref_prot = load_openeye_pdb(ref_prot, alt_loc=True)
        ## If alt locations are present in PDB file, set positions to highest
        ##  occupancy ALT
        alf = oechem.OEAltLocationFactory(ref_prot)
        if alf.GetGroupCount() != 0:
            alf.MakePrimaryAltMol(ref_prot)

    ## Split out protein components and align if requested
    if split_initial_complex:
        initial_prot = split_openeye_mol(initial_complex)["pro"]
    else:
        initial_prot = initial_complex
    if ref_prot is not None:
        if split_ref:
            ref_prot = split_openeye_mol(ref_prot)["pro"]
        initial_prot = superpose_molecule(ref_prot, initial_prot)[0]

    ## Add Hs to prep protein and ligand
    oechem.OEAddExplicitHydrogens(initial_prot)
    oechem.OEAddExplicitHydrogens(new_lig)

    ## Finally make new DesignUnit
    du = oechem.OEDesignUnit()
    oespruce.OEMakeDesignUnit(du, initial_prot, new_lig)
    assert du.HasProtein() and du.HasLigand()

    return du


def superpose_molecule(ref_mol, mobile_mol):
    """
    Superpose `mobile_mol` onto `ref_mol`.

    Parameters
    ----------
    ref_mol : oechem.OEGraphMol
        Reference molecule to align to.
    mobile_mol : oechem.OEGraphMol
        Molecule to align.

    Returns
    -------
    oechem.OEGraphMol
        New aligned molecule.
    float
        RMSD between `ref_mol` and `mobile_mol` after alignment.
    """

    ## Create object to store results
    aln_res = oespruce.OESuperposeResults()

    ## Set up superposing object and set reference molecule
    superpos = oespruce.OESuperpose()
    superpos.SetupRef(ref_mol)

    ## Perform superposing
    superpos.Superpose(aln_res, mobile_mol)
    print(f"RMSD: {aln_res.GetRMSD()}")

    ## Create copy of molecule and transform it to the aligned position
    mobile_mol_aligned = mobile_mol.CreateCopy()
    aln_res.Transform(mobile_mol_aligned)

    return mobile_mol_aligned, aln_res.GetRMSD()
