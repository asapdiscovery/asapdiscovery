from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.backend.openeye import save_openeye_pdb, oechem, oemol_to_pdb_string, pdb_string_to_oemol
from asapdiscovery.data.schema.target import Target
from asapdiscovery.data.util.dask_utils import FailureMode, dask_vmap
from asapdiscovery.modeling.modeling import find_component_chains
from pydantic import BaseModel
import MDAnalysis as mda
from MDAnalysis.lib.util import NamedStream
from io import StringIO
import warnings
import pymol2
import tempfile
import logging
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)





        # get the chain ID
        # chain_id = thisRes.GetChainID()
        # # if the chain ID is different from the previous one
        # if chain_id != prev_chain_id:
        #     # increment the chain ID
        #     chain_id = chr(ord(start_at) + 1)
        
        # thisRes.SetChainID(chain_id)
        # oechem.OEAtomSetResidue(atom, thisRes)

        

class SymmetryExpander(BaseModel):
    """
    Expand symmetry of a unit cell to include multiple copies.
    """

    expand_ligand: bool = False
    n_repeats: int = 8

    def expand(
        self,
        complexes: list[Complex],
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        **kwargs,
    ):

        return self._expand(
            complexes=complexes,
            use_dask=use_dask,
            dask_client=dask_client,
            failure_mode=failure_mode,
            **kwargs,
        )

    @dask_vmap(["complexes"], has_failure_mode=True)
    def _expand(
        self,
        complexes: list[Complex],
        failure_mode: str = "skip",
    ) -> list[Complex]:
        new_complexs = []
        for complex in complexes:
            # try:
                # target_oemol = complex.target.to_oemol()
                # if oechem.OEGetCrystalSymmetry(target_oemol) is None:
                #     raise ValueError("No crystal symmetry found in target")
                # new = oechem.OEMol()
                # oechem.OEExpandCrystalSymmetry(new, target_oemol, 10)
                # # combine
                # # chains = find_component_chains(target_oemol, sort_by="alphabetical")
                # # logger.info(f"Found chains: {chains}")
                # # last_chain = chains[-1]
                # # logger.info(f"Last chain: {last_chain}")
                # combined = oechem.OEGraphMol()
                # # hacky, forgive me 
                # universes = []
                # for i, mol in enumerate(new.GetConfs()):
                #     param = complex.target.crystal_symmetry
                #     if param is not None:
                #         p = oechem.OECrystalSymmetryParams(*param)
                #         oechem.OESetCrystalSymmetry(mol, p, True)
                #     save_openeye_pdb(mol, f"{i}_complex.pdb")
                #     oechem.OEAddMols(combined, mol)
                # save_openeye_pdb(combined, "combined.pdb")
                

                # c = Complex.from_pdb(
                #     "combined.pdb",
                #     target_kwargs={"target_name": "test"},
                #     ligand_kwargs={"compound_name": "test"},
                # )
                # new_complexs.append(c)
      
            try:
                p = pymol2.PyMOL()
                p.start()
                # round trip through                
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore"
                    )  # hides MDA RunTimeWarning that complains about string IO
                    u = mda.Universe(
                        NamedStream(StringIO(complex.target.data), "complex.pdb")
                    )

                    # check for a box
                    logger.info(f"symmetry expansion with box:  {u.trajectory.ts.dimensions}")
                    if u.trajectory.ts.dimensions is None:
                        raise ValueError("No box found")
                    elif not all(u.trajectory.ts.dimensions[:3]):
                        raise ValueError("Box has zero volume")


                # load each component into PyMOL
                p.cmd.read_pdbstr(complex.target.data, "protein_obj")
                p.cmd.read_pdbstr(
                    oemol_to_pdb_string(complex.ligand.to_oemol()), "lig_obj"
                )

                # remove some solvent stuff to prevent occasional clashes with neighbors
                p.cmd.remove("solvent")
                p.cmd.remove("inorganic")

                # reconstruct neighboring asymmetric units from the crystallographic experiment
                p.cmd.symexp(
                    "sym", "protein_obj", "(protein_obj)", self.n_repeats
                )  # do a big expansion just to be sure

                if self.expand_ligand:
                    p.cmd.symexp(
                        "sym", "lig_obj", "(lig_obj)", self.n_repeats
                    )  # do a big expansion just to be sure

                # select not the original unit
                p.cmd.select("not_ori", "!protein_obj and !lig_obj")
                # alter chain identifier to X
                p.cmd.alter("not_ori", "chain='X'")

                string = p.cmd.get_pdbstr(
                    "all", 0
                )  # writes all states, so should be able to handle multi-ligand
                p.stop()


                with open("fin.pdb", 'w') as f:
                    f.write(string)
                    cnew = Complex.from_pdb(
                        "fin.pdb",
                        target_kwargs={"target_name": "test"},
                        ligand_kwargs={"compound_name": "test"},
                    )

                u = mda.Universe("fin.pdb")
                cx = u.select_atoms("chainID X")
                # save
                cx.write("cx.pdb")
                ncx = u.select_atoms("not chainID X")
                # save
                ncx.write("ncx.pdb")
                new_complexs.append(cnew)

            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {complex.unique_name}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )
        return new_complexs