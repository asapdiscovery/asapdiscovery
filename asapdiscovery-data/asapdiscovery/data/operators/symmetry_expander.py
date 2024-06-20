from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.backend.openeye import save_openeye_pdb, oechem, oemol_to_pdb_string
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



def set_chain(mol, chain_code):
    hv = oechem.OEHierView(mol)
    for res in hv.GetResidues():
        res.GetOEResidue().SetExtChainID(str(chain_code))
    return mol

        

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
                oechem.OEExpandCrystalSymmetry(new, target_oemol, 40)
                # combine
                # chains = find_component_chains(target_oemol, sort_by="alphabetical")
                # logger.info(f"Found chains: {chains}")
                # last_chain = chains[-1]
                # logger.info(f"Last chain: {last_chain}")
                combined = oechem.OEMol()
                chain_code = 1
                for i, mol in enumerate(new.GetConfs()):
                    chain_code = chain_code +1
                    # set chains ID on the new molecule
                    # set_chain(mol, "X")
                    save_openeye_pdb(mol, f"tmp_{i}.pdb")
                    oechem.OEAddMols(combined, mol)

                t = Target.from_oemol(combined, target_name=complex.target.target_name, ids=complex.target.ids)
                c = Complex(target=t, ligand=complex.ligand, ligand_chain=complex.ligand_chain)
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