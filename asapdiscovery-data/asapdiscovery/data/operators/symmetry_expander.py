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
    n_repeats: int = 1

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
            try:
                target_oemol = complex.target.to_oemol()
                if oechem.OEGetCrystalSymmetry(target_oemol) is None:
                    raise ValueError("No crystal symmetry found in target")
                new = oechem.OEMol()
                oechem.OEExpandCrystalSymmetry(new, target_oemol, 10)
                # combine
                # chains = find_component_chains(target_oemol, sort_by="alphabetical")
                # logger.info(f"Found chains: {chains}")
                # last_chain = chains[-1]
                # logger.info(f"Last chain: {last_chain}")
                combined = oechem.OEGraphMol()
                # hacky, forgive me 
                universes = []
                for i, mol in enumerate(new.GetConfs()):
                    save_openeye_pdb(mol, f"{i}_complex.pdb")

                    oechem.OESetCrystalSymmetry(mol, oechem.OEGetCrystalSymmetry(target_oemol))
                    u = mda.Universe( NamedStream(StringIO(oemol_to_pdb_string(mol)), "complex.pdb"))
                    # set chain id
                    print(u.atoms.chainIDs)
                    # increment chain code
                    u.atoms.chainIDs = ["X"] * len(u.atoms)
                    universes.append(u)
                
                # merge
                m = mda.Merge(*[u.atoms for u in universes])
                print(m)
                print(m.atoms)
                m.atoms.write("tmp.pdb")

                c = Complex.from_pdb(
                    "tmp.pdb",
                    target_kwargs={"target_name": "test"},
                    ligand_kwargs={"compound_name": "test"},
                )
                new_complexs.append(c)
      
            # try:
            #     p = pymol2.PyMOL()
            #     p.start()
            #     # check if PDB has a box
            #     with warnings.catch_warnings():
            #         warnings.simplefilter(
            #             "ignore"
            #         )  # hides MDA RunTimeWarning that complains about string IO
            #         u = mda.Universe(
            #             NamedStream(StringIO(complex.target.data), "complex.pdb")
            #         )

            #         # check for a box
            #         logger.info(f"symmetry expansion with box:  {u.trajectory.ts.dimensions}")
            #         if u.trajectory.ts.dimensions is None:
            #             raise ValueError("No box found in PDB")
            #         elif not all(u.trajectory.ts.dimensions[:3]):
            #             raise ValueError("Box has zero volume")

            #     # load each component into PyMOL
            #     p.cmd.read_pdbstr(complex.target.data, "protein_obj")
            #     p.cmd.read_pdbstr(
            #         oemol_to_pdb_string(complex.ligand.to_oemol()), "lig_obj"
            #     )

            #     # remove some solvent stuff to prevent occasional clashes with neighbors
            #     p.cmd.remove("solvent")
            #     p.cmd.remove("inorganic")

            #     # reconstruct neighboring asymmetric units from the crystallographic experiment
            #     p.cmd.symexp(
            #         "sym", "protein_obj", "(protein_obj)", self.n_repeats
            #     )  # do a big expansion just to be sure

            #     if self.expand_ligand:
            #         p.cmd.symexp(
            #             "sym", "lig_obj", "(lig_obj)", self.n_repeats
            #         )  # do a big expansion just to be sure

            #     string = p.cmd.get_pdbstr(
            #         "all", 0
            #     )  # writes all states, so should be able to handle multi-ligand
            #     p.stop()

            #     tmp = tempfile.NamedTemporaryFile()
            #     with open(tmp.name, 'w') as f:
            #         f.write(string)
            #         cnew = Complex.from_pdb(
            #             tmp.name,
            #             target_kwargs={"target_name": "test"},
            #             ligand_kwargs={"compound_name": "test"},
            #         )

            #     new_complexs.append(cnew)

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