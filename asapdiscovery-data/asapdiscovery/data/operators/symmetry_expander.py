import logging
import warnings
from io import StringIO
from tempfile import NamedTemporaryFile

import MDAnalysis as mda
import pymol2
from asapdiscovery.data.backend.openeye import oemol_to_pdb_string
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.util.dask_utils import FailureMode, dask_vmap
from MDAnalysis.lib.util import NamedStream
from pydantic.v1 import BaseModel

logger = logging.getLogger(__name__)


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
                    logger.info(
                        f"symmetry expansion with box:  {u.trajectory.ts.dimensions}"
                    )
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

                temp = NamedTemporaryFile(suffix=".pdb")
                with open(temp.name, "w") as f:
                    f.write(string)
                    cnew = Complex.from_pdb(
                        temp.name,
                        target_kwargs=complex.target.dict(),
                        ligand_kwargs=complex.ligand.dict(),
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
