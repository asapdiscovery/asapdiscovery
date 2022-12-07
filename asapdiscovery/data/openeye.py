from openeye import oechem, oedocking, oegrid


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
