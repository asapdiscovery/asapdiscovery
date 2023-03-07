import argparse
import multiprocessing as mp
from pathlib import Path
from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    openeye_perceive_residues,
    save_openeye_pdb,
    split_openeye_design_unit,
    split_openeye_mol,
)
from openmm.app import PDBFile, PDBxFile
from asapdiscovery.docking.modeling import align_receptor
import yaml
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.docking.modeling import mutate_residues
from asapdiscovery.modeling.modeling import spruce_protein
from asapdiscovery.data.openeye import oechem


def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to downloaded cif1 files.",
    )
    parser.add_argument(
        "-r",
        "--ref_prot",
        default="../tests/prep_mers_rcsb/inputs/reference.pdb",
        type=str,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
    )

    ## Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    ## Model-building arguments
    parser.add_argument(
        "-l",
        "--loop_db",
        default="/Users/alexpayne/Scientific_Projects/mers-drug-discovery/spruce_bace.loop_db",
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        default="../../../../metadata/mpro_mers_seqres.yaml",
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=True,
        help="If true, generate design units with only the protein in them",
    )
    parser.add_argument(
        "--log_file",
        default="prep_proteins_log.txt",
        help="Path to high level log file.",
    )
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    return parser.parse_args()


def prep_mp(cifpath, output, loop_db, ref_prot, seqres_yaml):
    du_fn = output / f"{cifpath.stem}-prepped_receptor_0.oedu"
    if du_fn.exists():
        print(f"Already made {du_fn}!")
        return

    print("Loading cif and writing to pdb file")

    cif = PDBxFile(str(cifpath))

    outfile = output / f"{cifpath.stem}-openmm.pdb"

    ## the keep ids flag is critical to make sure the residue numbers are correct
    with open(outfile, "w") as f:
        PDBFile.writeFile(cif.topology, cif.positions, f, keepIds=True)

    print("Loading pdb to OpenEye")
    prot = load_openeye_pdb(str(outfile))

    print("Aligning to ref")

    ref_path = Path(ref_prot)
    prot = align_receptor(
        initial_complex=prot,
        ref_prot=ref_path.as_posix(),
        dimer=True,
        split_initial_complex=True,
        mobile_chain="A",  # TODO: make this not hardcoded? not sure what logic to use though
        ref_chain="A",
    )
    # aligned = str(output / f"{cifpath.stem}-01.pdb")
    # save_openeye_pdb(prot, aligned)

    print("Preparing Sprucing options")
    loop_path = Path(loop_db)

    seqres_path = Path(seqres_yaml)
    with open(seqres_path) as f:
        seqres_dict = yaml.safe_load(f)
    seqres = seqres_dict["SEQRES"]

    res_list = seqres_to_res_list(seqres)
    seqres = " ".join(res_list)

    print("Making mutations")

    prot = mutate_residues(prot, res_list, place_h=True)

    print("Sprucing protein")

    du = spruce_protein(
        initial_prot=prot,
        seqres=seqres,
        loop_db=str(loop_path),
        return_du=True,
    )

    if type(du) == oechem.OEDesignUnit:

        print("Saving Design Unit")

        du_fn = output / f"{cifpath.stem}-prepped_receptor_0.oedu"
        oechem.OEWriteDesignUnit(str(du_fn), du)

        print("Saving PDB")

        prot = oechem.OEGraphMol()
        du.GetProtein(prot)

        ## Add SEQRES entries if they're not present
        if (not oechem.OEHasPDBData(prot, "SEQRES")) and seqres:
            for seqres_line in seqres.split("\n"):
                if seqres_line != "":
                    oechem.OEAddPDBData(prot, "SEQRES", seqres_line[6:])

        prot_fn = output / f"{cifpath.stem}-prepped_receptor_0.pdb"
        save_openeye_pdb(prot, str(prot_fn))

    elif type(du) == oechem.OEGraphMol:
        print("Design Unit preparation failed. Saving spruced protein")
        prot_fn = output / f"{cifpath.stem}-failed-spruced.pdb"
        save_openeye_pdb(du, str(prot_fn))


def main():
    args = get_args()

    inputs = Path(args.structure_dir)
    cifpaths = inputs.glob("*-assembly1.cif")
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True)

    mp_args = [
        [cifpath, output, args.loop_db, args.ref_prot, args.seqres_yaml]
        for cifpath in cifpaths
    ]
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    print(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
