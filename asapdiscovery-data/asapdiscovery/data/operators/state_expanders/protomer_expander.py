import os
import subprocess
import tempfile
from typing import Literal

from asapdiscovery.data.backend.openeye import (
    load_openeye_sdfs,
    oechem,
    oequacpac,
    save_openeye_sdfs,
)
from asapdiscovery.data.operators.state_expanders.state_expander import (
    StateExpanderBase,
)
from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import Field


class ProtomerExpander(StateExpanderBase):
    """
    Expand a molecule to protomers using OpenEye reasonable protomer state enumeration.

    Note:
        The input molecule is included in the output.
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
                if protomer_ligand.fixed_inchikey != parent_ligand.fixed_inchikey:
                    # only add tags to new microstates of the input molecule
                    protomer_ligand.set_expansion(
                        parent=parent_ligand, provenance=provenance
                    )
                    expanded_states.append(protomer_ligand)
                else:
                    expanded_states.append(parent_ligand)
            # add the parent if it is not present.
            if parent_ligand not in expanded_states:
                expanded_states.append(parent_ligand)

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
        if schrodinger_folder is None:
            raise RuntimeError(
                "Epik enumerator requires the path to the schrodinger software to be set as the "
                "SCHRODINGER environment variable."
            )
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
                convert_cmd + " input.sdf input.mae",
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
                convert_cmd + " output.mae output.sdf",
                shell=True,
                stdout=log,
                stderr=log,
                check=True,
            )
        oe_mols = load_openeye_sdfs(sdf_fn="output.sdf")
        # parse into ligand objects
        expanded_ligands = [Ligand.from_oemol(oemol) for oemol in oe_mols]
        return expanded_ligands

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
         All input molecules are scored by Epik and have the values stored in Ligand.tags. Only new molecules will
         have an expansion tag however. For example ethane would receive a score but no expansion tag.

        Parameters
        ----------
        ligands: The list of ligands whose states should be expanded.

        Returns
        -------
            A list of expanded ligand states.
        """
        # store where we are as we run epik in a tempdir
        home = os.getcwd()
        # calculate it once as its expensive to call epik every time
        provenance = self.provenance()

        # as epic runs on all molecules we need to keep track of the parent by tagging it
        parents_by_inchikey = {}
        for lig in ligands:
            # store the parent inchi key as a tag which will be included in the sdf file
            fixed_inchikey = lig.fixed_inchikey
            lig.set_SD_data({"parent": fixed_inchikey})
            parents_by_inchikey[fixed_inchikey] = lig

        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)

            # create the mae file
            self._prepare_ligands(ligands=ligands)

            # call epik
            self._call_epik()

            # convert the ligands to sdf, epik tags are automatically picked up and stored
            expanded_ligands = self._extract_ligands()

            # move back to the home dir
            os.chdir(home)

        # set the expansion tag only for new microstate ligands
        for ligand in expanded_ligands:
            # do not set the expansion tag if the molecule is the same as the parent and has a score of 0
            state_pentalty = float(ligand.tags["r_epik_State_Penalty"])
            if ligand.tags["parent"] == ligand.fixed_inchikey and state_pentalty == 0:
                continue

            parent = parents_by_inchikey[ligand.tags["parent"]]
            ligand.set_expansion(parent=parent, provenance=provenance)

        return expanded_ligands
