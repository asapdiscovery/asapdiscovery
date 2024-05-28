from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.backend.openeye import oemol_to_pdb_string
from asapdiscovery.data.util.dask_utils import FailureMode, dask_vmap
from pydantic import BaseModel
import MDAnalysis as mda
from MDAnalysis.lib.util import NamedStream
from io import StringIO
import warnings
import pymol2
import tempfile
import logging

logger = logging.getLogger(__name__)


class SymmetryExpander(BaseModel):
    """
    Expand symmetry of a unit cell to include multiple copies.
    """

    expand_ligand: bool = False

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
                p = pymol2.PyMOL()
                p.start()
                # check if PDB has a box
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore"
                    )  # hides MDA RunTimeWarning that complains about string IO
                    u = mda.Universe(
                        NamedStream(StringIO(complex.target.data), "complex.pdb")
                    )

                    # check for a box
                    has_box = all(u.trajectory.ts.dimensions[:3])
                    if not has_box:
                        raise ValueError(
                            "Cannot perform expansion as Complex does not have bounding box"
                        )
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
                    "sym", "protein_obj", "(protein_obj)", 8
                )  # do a big expansion just to be sure

                if self.expand_ligand:
                    p.cmd.symexp(
                        "sym", "lig_obj", "(lig_obj)", 8
                    )  # do a big expansion just to be sure

                string = p.cmd.get_pdbstr(
                    "all", 0
                )  # writes all states, so should be able to handle multi-ligand
                p.stop()

                # tmp = tempfile.NamedTemporaryFile()
                # with open(tmp.name, 'w') as f:
                with open("tst.pdb", "w") as f:

                    f.write(string)
                    cnew = Complex.from_pdb(
                        "tst.pdb",
                        target_kwargs={"target_name": "test"},
                        ligand_kwargs={"compound_name": "test"},
                    )

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
