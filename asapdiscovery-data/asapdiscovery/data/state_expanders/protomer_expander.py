import os
import subprocess
from typing import Literal

from asapdiscovery.data.openeye import (
    load_openeye_sdfs,
    oechem,
    oequacpac,
    save_openeye_sdfs,
)
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.state_expander import StateExpanderBase
from pydantic import Field


class ProtomerExpander(StateExpanderBase):
    """
    Expand a molecule to protomers
    """

    expander_type: Literal["ProtomerExpander"] = "ProtomerExpander"

    def _provenance(self) -> dict[str, str]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "quacpac": oequacpac.OEQuacPacGetVersion(),
        }

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        expanded_states = []
        provenance = self.provenance()
        for parent_ligand in ligands:
            oemol = parent_ligand.to_oemol()
            for protomer in oequacpac.OEGetReasonableProtomers(oemol):
                fmol = oechem.OEMol(protomer)
                # copy the ligand properties over to the new molecule, we may want to have more fine grained control over this
                # down the track.
                protomer_ligand = Ligand.from_oemol(fmol, **parent_ligand.dict())
                protomer_ligand.set_expandsion(
                    parent=parent_ligand, provenance=provenance
                )
                expanded_states.append(protomer_ligand)

        return expanded_states


class EpikExpander(StateExpanderBase):
    """
    Expand the protomer and tautomeric states of a molecule using epik and capture the state penalties.

    Note:
        The method assumes you have schrodinger software installed and the path to the software is exported as a
        environment variable named SCHRODINGER.
    """

    expander_type: Literal["EpikExpander"] = "EpikExpander"

    ph: float = Field(
        7.3,
        description="The ph that should be used when calculating the state penalty.",
    )

    def _create_cmd(self, *programs: str) -> str:
        """
        Create a command which can be used to call some SCHRODINGER software
        Returns
        -------
            The string which can be passed to subprocess to call epik
        """
        # create a path to epik
        schrodinger_folder = os.getenv("SCHRODINGER")
        epik = os.path.join(schrodinger_folder, *programs)
        return epik

    def _provenance(self) -> dict[str, str]:
        """
        Run epik to get the version info.
        Returns
        -------
            The version of epik used.
        """
        epik_cmd = self._create_cmd("epik")
        # call epik to get the version info
        output = subprocess.check_output([epik_cmd, "-v"])
        for line in output.decode("utf-8").split("\n"):
            if "Epik version" in line:
                version = line.split()[-1]
                break
        else:
            version = "unknown"

        return {
            "epik": version,
        }

    def _prepare_ligands(self, ligands: list[Ligand]):
        """
        Convert the list of Ligands to a SCHRODINGER mae file before running with Epik.
        """
        oe_ligands = [ligand.to_oemol() for ligand in ligands]
        save_openeye_sdfs(oe_ligands, "input.sdf")
        convert_cmd = self._create_cmd("utilities", "structconvert")
        with open("structconvert.log", "w") as log:
            subprocess.run(
                convert_cmd + "input.sdf input.mae",
                shell=True,
                stdout=log,
                stderr=log,
                check=True,
            )

    def _extract_ligands(self) -> list[Ligand]:
        """
        Extract the state expanded ligands from the Epik output file.
        Returns
        -------
            A list of expanded state ligands.
        """
        convert_cmd = self._create_cmd("utilities", "structconvert")
        with open("structconvert.log", "w") as log:
            subprocess.run(
                convert_cmd + "output.mae output.sdf",
                shell=True,
                stdout=log,
                stderr=log,
                check=True,
            )
        oe_mols = load_openeye_sdfs(sdf_fn="output.sdf")
        # parse into ligand objects
        return [Ligand.from_oemol(oe_mol) for oe_mol in oe_mols]

    def _call_epik(self):
        """Call Epik on the local ligands file."""
        import numpy as np

        epik_command = self._create_cmd("epik")
        min_population = np.exp(-6)
        epik_command += f" -WAIT -ms 16 -ph {self.ph} -p {min_population} -imae input.mae -omae output.mae"
        with open("epik_log.log", "w") as log:
            subprocess.run(epik_command, shell=True, stdout=log, stderr=log, check=True)

    def _expand(self, ligands: list[Ligand]) -> list[Ligand]:
        """
        Expand the protomers and tautomers of the input molecules using Epik and calculate the state penalty.

        Note:
        The input molecules are included in the output.

        Parameters
        ----------
        ligands: The list of ligands who's states should be expanded.

        Returns
        -------
            A list of expanded ligand states.
        """
        # calculate it once as its expensive to call epik every time
        provenance = self.provenance()

        # as epic runs on all molecules we need to keep track of the parent by tagging it
        parents_by_inchikey = {}
        for lig in ligands:
            # store the parent inchi key
            fixed_inchikey = lig.fixed_inchikey
            lig.set_SD_data({"parent": fixed_inchikey})
            parents_by_inchikey[fixed_inchikey] = lig

        # create the mae file
        self._prepare_ligands(ligands=ligands)

        # call epik
        self._call_epik()

        # convert the ligands to sdf
        expanded_ligands = self._extract_ligands()

        # set the expansion tag for each molecule
        for ligand in expanded_ligands:
            parent = parents_by_inchikey[ligand.tags["parent"]]
            # extract the epik data
            state_info = {
                (key, value) for key, value in ligand.tags.items() if "epik" in key
            }
            ligand.set_expansion(
                parent=parent, provenance=provenance, state_information=state_info
            )

        return expanded_ligands
