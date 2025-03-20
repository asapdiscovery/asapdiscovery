import abc
import warnings
from typing import Any, Literal

from asapdiscovery.alchemy.schema.base import _SchemaBase
from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import Field
from tqdm import tqdm


class _BaseChargeMethod(_SchemaBase, abc.ABC):
    type: Literal["_BaseChargeMethod"] = "_BaseChargeMethod"

    def provenance(self) -> dict[str, str]:
        """
        Get the provenance of the software and settings used to generate partial charges for the molecule.

        Returns:
            A dict of the charge generation method with the software versions.
        """
        data = {"protocol": self.dict(), "provenance": self._provenance()}
        return data

    @abc.abstractmethod
    def _provenance(self) -> dict[str, Any]:
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
        self, ligands: list[Ligand], processors: int = 1
    ) -> list[Ligand]:
        return self._generate_charges(ligands=ligands, processors=processors)


class OpenFFCharges(_BaseChargeMethod):
    type: Literal["OpenFFCharges"] = "OpenFFCharges"

    charge_method: Literal["am1bccelf10", "am1bcc"] = Field(
        "am1bccelf10",
        description="The OpenFF toolkit supported " "charging method to use.",
    )

    def _provenance(self) -> dict[str, Any]:
        import openff.toolkit

        provenance = {"openff.toolkit": openff.toolkit.__version__}
        if self.charge_method == "am1bccelf10":
            from openeye import oeomega, oequacpac

            provenance["oeomega"] = str(oeomega.OEOmegaGetVersion())
            provenance["oequacpac"] = str(oequacpac.OE_OEQUACPAC_VERSION)
        else:
            import rdkit
            from openff.utilities import get_ambertools_version

            provenance["rdkit"] = rdkit.__version__
            provenance["ambertools"] = get_ambertools_version()
        return provenance

    def _charge_molecule(self, ligand: Ligand) -> Ligand:
        """Generate charges for the molecule using the openff toolkit."""
        from openff.toolkit import Molecule

        try:
            off_mol = Molecule.from_rdkit(ligand.to_rdkit())
            off_mol.assign_partial_charges(partial_charge_method=self.charge_method)
            # fake the creation of the rdkit double property list
            charges = " ".join([str(e) for e in off_mol.partial_charges.m])
            ligand.tags["atom.dprop.PartialCharge"] = charges
            return True, ligand, None
        except Exception as e:
            return False, ligand, e

    def _generate_charges(
        self,
        ligands: list[Ligand],
        processors: int = 1,
    ) -> list[Ligand]:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        provenance = self.provenance()
        charged_ligands = []
        failed_ligands = []

        if processors > 1:
            progressbar = tqdm(total=len(ligands))
            with ProcessPoolExecutor(max_workers=processors) as pool:
                work_list = [
                    pool.submit(self._charge_molecule, ligand) for ligand in ligands
                ]
                for work in as_completed(work_list):
                    succ, result_ligand, err_code = work.result()
                    if succ:
                        charged_ligands.append(result_ligand)
                    else:
                        failed_ligands.append(result_ligand)
                        warnings.warn(
                            f"Ligand charging failed for ligand {result_ligand.compound_name}:{result_ligand.smiles} with exception: {err_code}"
                        )

                    progressbar.update(1)

        else:
            for ligand in tqdm(ligands, total=len(ligands)):
                succ, result_ligand, err_code = self._charge_molecule(ligand=ligand)
                if succ:
                    charged_ligands.append(result_ligand)
                else:
                    failed_ligands.append(result_ligand)
                    warnings.warn(
                        f"Ligand charging failed for ligand {result_ligand.compound_name}:{result_ligand.smiles} with exception: {err_code}"
                    )

        for ligand in charged_ligands:
            # stamp how the charges were made
            ligand.charge_provenance = provenance

        return charged_ligands, failed_ligands
