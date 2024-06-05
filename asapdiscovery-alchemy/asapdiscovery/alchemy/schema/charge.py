from asapdiscovery.alchemy.schema.base import _SchemaBase
from pydantic import Field
from typing import Literal, Any
import abc
from asapdiscovery.data.schema.ligand import Ligand


class _BaseChargeMethod(_SchemaBase, abc.ABC):

    type: Literal["_BaseChargeMethod"] = "_BaseChargeMethod"

    @abc.abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return the provenance for this pose generation method."""
        ...

    @abc.abstractmethod
    def _generate_charges(
            self,
            ligands: list[Ligand],
            processors: int = 1,
    ) -> list[Ligand]:
        """The main worker method which should be used to generate charges for the ligands."""
        ...

    def generate_charges(
            self,
            ligands: list[Ligand],
            processors: int = 1
    ) -> list[Ligand]:
        return self._generate_charges(ligands=ligands, processors=processors)


class OpenFFCharges(_BaseChargeMethod):

    type: Literal["OpenFFCharges"] = "OpenFFCharges"

    charge_method: Literal["am1bccelf10", "am1bcc"] = Field("am1bccelf10", description="The OpenFF toolkit supported "
                                                                                       "charging method to use.")

    def provenance(self) -> dict[str, Any]:
        import openff.toolkit
        provenance = {"openff.toolkit": openff.toolkit.__version__}
        if self.charge_method == "am1bccelf10":
            from openeye import oequacpac, oeomega
            provenance["oeomega"] = oeomega.OEOmegaGetVersion()
            provenance["oequacpac"] = oequacpac.OE_OEQUACPAC_VERSION
        else:
            import rdkit
            from openff.utilities import get_ambertools_version
            provenance["rdkit"] = rdkit.__version__
            provenance["ambertools"] = get_ambertools_version()
        return provenance

    def _charge_molecule(self, ligand: Ligand) -> Ligand:
        """Generate charges for the molecule using the openff toolkit."""
        from openff.toolkit import Molecule
        off_mol = Molecule.from_rdkit(ligand.to_rdkit())
        off_mol.assign_partial_charges(partial_charge_method=self.charge_method)
        # fake the creation of the rdkit double property list
        charges = " ".join([str(e) for e in off_mol.partial_charges.m])
        ligand.tags["atom.dprop.PartialCharge"] = charges
        return ligand

    def _generate_charges(
            self,
            ligands: list[Ligand],
            processors: int = 1,
    ) -> list[Ligand]:
        from openff.toolkit import Molecule
        from concurrent.futures import ProcessPoolExecutor, as_completed

        charge_method = self.dict()
        charge_method["provenance"] = self.provenance()
        charged_ligands = []

        if processors > 1:
            with ProcessPoolExecutor(max_workers=processors) as pool:
                work_list = [
                    pool.submit(
                        self._charge_molecule,
                        ligand
                    )
                    for ligand in ligands
                ]
                for work in as_completed(work_list):
                    result_ligand = work.result()
                    charged_ligands.append(result_ligand)

        else:
            for ligand in ligands:
                charged_ligands.append(self._charge_molecule(ligand=ligand))

        for ligand in charged_ligands:
            # stamp how the charges were made
            ligand.tags["charge_generation"] = charge_method

        return charged_ligands
