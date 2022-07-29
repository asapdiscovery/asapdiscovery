import argparse
import os
from openeye import oechem, oespruce

################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-target", required=True, help="PDB file for target protein."
    )
    parser.add_argument("-ref", help="PDB file for reference protein")
    parser.add_argument("-loop", help="SPRUCE loop database.")
    parser.add_argument("-o", required=True, help="Output directory.")

    return parser.parse_args()


def main():
    args = get_args()

    ## Load target molecule
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA,
    )
    ifs.open(args.target)
    in_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, in_mol)
    ifs.close()

    if args.ref is not None:
        ## Load reference molecule
        ifs.open(args.ref)
        ref_mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, ref_mol)
        ifs.close()

        ##Extract and align protein
        bio_opts = oespruce.OEBioUnitExtractionOptions()
        bio_opts.SetSuperpose(True)

        biounits = oespruce.OEExtractBioUnits(in_mol, ref_mol, bio_opts)
        in_mol = list(biounits)[0]

    ## Set up options for building DesignUnits
    opts = oespruce.OEMakeDesignUnitOptions()
    # opts.SetBioUnitExtractionOptions(bio_opts)
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetBuildTails(
        False
    )
    opts.GetPrepOptions().GetBuildOptions().GetLoopBuilderOptions().SetLoopDBFilename(
        args.loop
    )
    opts.GetPrepOptions().GetEnumerateSitesOptions().SetRestrictToRefSite(True)

    ## Build DesignUnits
    design_units = oespruce.OEMakeDesignUnits(
        in_mol, oespruce.OEStructureMetadata(), opts
    )
    design_units = list(design_units)
    out_base_du = (
        f"{args.o}/"
        f"{os.path.splitext(os.path.basename(args.target))[0]}_{{}}.oedu"
    )
    out_base = (
        f"{args.o}/"
        f"{os.path.splitext(os.path.basename(args.target))[0]}_du_protein_{{}}.pdb"
    )
    ofs = oechem.oemolostream()
    ofs.SetFlavor(
        oechem.OEFormat_PDB,
        oechem.OEOFlavor_PDB_Default,
    )
    prot_mol = oechem.OEGraphMol()

    design_units = list(design_units)
    for i, du in enumerate(design_units):
        print(i)
        ## Save the DesignUnit object
        oechem.OEWriteDesignUnit(out_base_du.format(i), du)

        ## Save the protein as a PDB file
        prot_mol.Clear()
        du.GetProtein(prot_mol)

        ofs.open(out_base.format(i))
        oechem.OEWriteMolecule(ofs, prot_mol)
        ofs.close()


if __name__ == "__main__":
    main()
