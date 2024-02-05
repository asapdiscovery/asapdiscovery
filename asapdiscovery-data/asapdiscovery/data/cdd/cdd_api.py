import json
from typing import Optional

from asapdiscovery.data.schema_v2.experimental import ExperimentalCompoundData
from asapdiscovery.data.service_confing import CDDSettings
from asapdiscovery.data.web_utils import _BaseWebAPI


class CDDAPI(_BaseWebAPI):
    """"""

    def __init__(self, url: str, api_version: str, api_key: str, vault: str):
        super().__init__(url=url, api_version=api_version, api_key=api_key)
        # now fix the url str
        self.api_url += f"/vaults/{vault}/"

    @classmethod
    def token_name(cls) -> str:
        return "X-CDD-token"

    @classmethod
    def from_settings(cls, settings: CDDSettings):
        return cls(
            url=settings.CDD_API_URL,
            api_version=settings.CDD_API_VERSION,
            api_key=settings.CDD_API_KEY,
            vault=settings.CDD_VAULT_NUMBER,
        )

    def get_molecule(self, smiles: str) -> Optional[dict]:
        """
        Search for a molecule in the CDD vault.

        Args:
            smiles: The smiles of the molecule to search for.

        Returns:
            A dictionary of the molecule details if present in CDD else None.

        """
        mol_data = {
            "structure": smiles,
            "no_structures": "true",
            "structure_search_type": "exact",
        }
        result = self._session.get(url=self.api_url + "molecules/", json=mol_data)
        result_data = json.loads(result.content.decode())
        if result_data["count"] == 0:
            return None
        else:
            # should only be one molecule but maybe we should check?
            return result_data["objects"][0]

    def get_protocol(
        self, molecule_id: int, protocol_name: Optional[str] = None
    ) -> list[dict]:
        """
        Search for a specific protocol performed on the query molecule.

        Args:
            molecule_id: The CDD id of the molecule whos protocols we want to gather.
            protocol_name: The name of the protocol to search for, if not provided all protocols will be pulled.

        Returns:
            A list of protocols associated with this molecule
        """
        protocol_data = {"molecules": [molecule_id]}
        if protocol_name is not None:
            protocol_data["names"] = [protocol_name]
        result = self._session.get(url=self.api_url + "protocols", json=protocol_data)
        result_data = json.loads(result.content.decode())
        return result_data["objects"]

    def get_readout_row(self, molecule_id: int, protocol: int) -> Optional[dict]:
        """
        Get the readout data for a specific protocol performed on a molecule. This is used to pull the pIC50_Mean.

        Args:
            molecule_id: The CDD id of the molecule to get the values for.
            protocol: The id of the protocol to use in the search.

        Returns:
            A dictionary of the readout data matching the search. The actual values are stored under `readouts`.
        """
        readout_data = {
            "protocols": [protocol],
            "molecules": [molecule_id],
            # this is hard coded to get the pIC50_Mean
            "type": ["molecule_protocol_aggregate_row"],
        }
        result = self._session.get(url=self.api_url + "readout_rows", json=readout_data)
        result_data = json.loads(result.content.decode())
        if result_data["count"] == 0:
            return None
        else:
            return result_data["objects"][0]

    def get_pic50(
        self, smiles: str, protocol_name: str
    ) -> Optional[ExperimentalCompoundData]:
        """
        A convenience method which wraps the required function calls to gather the pIC50 from the CDD for the given
        molecule calculated as part of the named protocol.

        Notes:
            The uncertainty is returned as 0 if the experimental protocol has a single value.

        Args:
            smiles: The smiles of the molecule we want the pIC50 for.
            protocol_name: The name of the protocol we want the pIC50 for.

        Returns:
            A ExperimentalCompoundData object with the pIC50_Mean and uncertainty if found else None.
        """
        # get the molecule id
        molecule = self.get_molecule(smiles=smiles)
        # check we found the molecule
        if molecule is None:
            return molecule
        # look for the protocol, we expect a single result as we search for a named protocol
        # if we don't find anything for the molecule return None
        protocols = self.get_protocol(
            molecule_id=molecule["id"], protocol_name=protocol_name
        )
        if protocols:
            protocol = protocols[0]
        else:
            return None
        # make sure the pic50 was calculated
        pic50_id = None
        for readout_def in protocol["readout_definitions"]:
            if readout_def["name"] == "pIC50_Mean":
                pic50_id = readout_def["id"]
                break
        if pic50_id is None:
            return None
        # now search for the result, this should always be present if the above protocol has a result
        readout_data = self.get_readout_row(
            molecule_id=molecule["id"], protocol=protocol["id"]
        )
        # extract the results
        pic50_data = readout_data["readouts"][str(pic50_id)]
        pic50 = pic50_data["value"]
        # workout if we have an error
        if pic50_data["note"] == "(n=1)":
            pic50_uncertainty = 0
        else:
            pic50_uncertainty = float(pic50_data["note"].split()[1])
        # gather the data and details
        experimental_result = ExperimentalCompoundData(
            compound_id=molecule["name"],
            smiles=smiles,
            achiral=molecule["batches"][0]["stereochem_data"] == "Achiral",
            date_created=molecule["batches"][0]["created_at"],
            experimental_data={
                "pIC50_Mean": pic50,
                "pIC50_Uncertainty": pic50_uncertainty,
                "protocol_name": protocol_name,
                "protocol_id": protocol["id"],
            },
        )
        return experimental_result
