from openeye import oechem, oedocking, oegrid


def load_openeye_pdb(pdb_fn):
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA,
    )
    ifs.open(pdb_fn)
    in_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, in_mol)
    ifs.close()

    return in_mol


def load_openeye_sdf(sdf_fn):
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    ifs.open(sdf_fn)
    coords_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, coords_mol)
    ifs.close()

    return coords_mol


def load_openeye_sdfs(sdf_fn):
    """
    Parameters
    ----------
    sdf_fn: str
        SDF file path
    Returns
    -------
    list of OEGraphMol objects
    """
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    cmpd_list = []
    if ifs.open(sdf_fn):
        for mol in ifs.GetOEGraphMols():
            cmpd_list.append(mol.CreateCopy())
    else:
        oechem.OEThrow.Fatal(f"Unable to open {sdf_fn}")
    ifs.close()

    return cmpd_list


def split_openeye_mol(complex_mol: oechem.OEMolBase):
    ## Test splitting
    lig_mol = oechem.OEGraphMol()
    prot_mol = oechem.OEGraphMol()
    water_mol = oechem.OEGraphMol()
    oth_mol = oechem.OEGraphMol()

    ## Make splitting split out covalent ligands
    ## TODO: look into different covalent-related options here
    opts = oechem.OESplitMolComplexOptions()
    opts.SetSplitCovalent(True)
    opts.SetSplitCovalentCofactors(True)
    print(
        oechem.OESplitMolComplex(
            lig_mol, prot_mol, water_mol, oth_mol, complex_mol
        )
    )

    print(
        complex_mol.NumAtoms(),
        lig_mol.NumAtoms(),
        prot_mol.NumAtoms(),
        water_mol.NumAtoms(),
        oth_mol.NumAtoms(),
    )
    return {
        "complex": complex_mol,
        "lig": lig_mol,
        "pro": prot_mol,
        "water": water_mol,
        "other": oth_mol,
    }


def get_ligand_rmsd_from_pdb_and_sdf(
    ref_path, mobile_path, fetch_docking_results=True
):
    ref_pdb = load_openeye_pdb(ref_path)
    ref = split_openeye_mol(ref_pdb)["lig"]
    mobile = load_openeye_sdf(mobile_path)

    for a in mobile.GetAtoms():
        if a.GetAtomicNum() == 1:
            mobile.DeleteAtom(a)

    rmsd = oechem.OERMSD(ref, mobile)

    return_dict = {"rmsd": rmsd}

    if fetch_docking_results:
        return_dict["posit"] = oechem.OEGetSDData(mobile, "POSIT::Probability")
        return_dict["chemgauss"] = oechem.OEGetSDData(mobile, "Chemgauss4")

    return return_dict


def save_openeye_design_unit(du, lig=None, lig_title=None):
    """
    Parameters
    ----------
    du : oechem.OEDesignUnit
        Design Unit to be saved

    Returns
    -------
    lig : oechem.OEGraphMol
        OE object containing ligand
    protein : oechem.OEGraphMol
        OE object containing only protein
    complex : oechem.OEGraphMol
        OE object containing ligand + protein
    """
    prot = oechem.OEGraphMol()
    complex = oechem.OEGraphMol()
    du.GetProtein(prot)
    if not lig:
        lig = oechem.OEGraphMol()
        du.GetLigand(lig)

    ## Set ligand title, useful for the combined sdf file
    if lig_title:
        lig.SetTitle(f"{lig_title}")

    ## Give ligand atoms their own chain "L" and set the resname to "LIG"
    residue = oechem.OEAtomGetResidue(next(iter(lig.GetAtoms())))
    residue.SetChainID("L")
    residue.SetName("LIG")
    residue.SetHetAtom(True)
    for atom in list(lig.GetAtoms()):
        oechem.OEAtomSetResidue(atom, residue)

    ## Combine protein and ligand and save
    ## TODO: consider saving water as well
    oechem.OEAddMols(complex, prot)
    oechem.OEAddMols(complex, lig)

    ## Clean up PDB info by re-perceiving, perserving chain ID, residue number, and residue name
    preserve = (
        oechem.OEPreserveResInfo_ChainID
        | oechem.OEPreserveResInfo_ResidueNumber
        | oechem.OEPreserveResInfo_ResidueName
    )
    oechem.OEPerceiveResidues(prot, preserve)
    return lig, prot, complex


def save_receptor_grid(du_fn, out_fn):
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(du_fn, du)
    # oedocking.OEMakeReceptor(du)
    oegrid.OEWriteGrid(
        out_fn,
        oegrid.OEScalarGrid(du.GetReceptor().GetNegativeImageGrid()),
    )
