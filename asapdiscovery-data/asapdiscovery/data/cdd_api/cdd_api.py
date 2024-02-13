import json
from typing import Optional

import pandas
from asapdiscovery.data.services_config import CDDSettings
from asapdiscovery.data.web_utils import _BaseWebAPI


class CDDAPI(_BaseWebAPI):
    """
    An interface to the CDD JSON API which allows you to search for molecules protocols and readouts like IC50.
    """

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
            molecule_id: The CDD id of the molecule who's protocols we want to gather.
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

    def get_readout_row(
        self, molecule_id: int, protocol: int, types: Optional[list[str]] = None
    ) -> Optional[dict]:
        """
        Get the readout data for a specific protocol performed on a molecule. This is used to pull the pIC50_Mean.

        Args:
            molecule_id: The CDD id of the molecule to get the values for.
            protocol: The id of the protocol to use in the search.
            types: A list of readout types to pull the results for.

        Returns:
            A dictionary of the readout data matching the search. The actual values are stored under `readouts`.
        """
        readout_data = {
            "protocols": [protocol],
            "molecules": [molecule_id],
        }
        if types is not None:
            readout_data["type"] = types
        result = self._session.get(url=self.api_url + "readout_rows", json=readout_data)
        result_data = json.loads(result.content.decode())
        if result_data["count"] == 0:
            return None
        else:
            return result_data["objects"]

    def get_ic50_data(self, smiles: str, protocol_name: str) -> Optional[list[dict]]:
        """
        A convenience method which wraps the required function calls to gather the raw ic50 data from the CDD for the
        given molecule calculated as part of the named protocol.

        Args:
            smiles: The smiles of the molecule we want the IC50 for.
            protocol_name: The name of the protocol we want the IC50 for.

        Returns:
            A list of dictionaries containing the IC50 values along with upper and lower CI and curve class for each
            batch measurement on the molecule performed as part of the given protocol.

        Notes:
            Returns None if no IC50 data can be collected.
            Returns all batch IC50 data for a molecule.
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
        # define the readouts we want to find
        required_data = {
            "IC50": None,
            "IC50 CI (Lower)": None,
            "IC50 CI (Upper)": None,
            "Curve class": None,
        }
        for readout_def in protocol["readout_definitions"]:
            if (readout_name := readout_def["name"]) in required_data:
                # gather the id of result for this readout
                required_data[readout_name] = readout_def["id"]
        # if any of the data is missing return
        if None in required_data:
            return None

        # pull down all batch readouts for this protocol and extract the data
        readout_data = self.get_readout_row(
            molecule_id=molecule["id"],
            protocol=protocol["id"],
            types=["batch_run_aggregate_row"],
        )
        # extract the results
        ic50_data = []
        for readout in readout_data:
            try:
                batch_data = {
                        f"{protocol_name}: {key} {'(µM)' if 'IC50' in key else ''}":
                        readout["readouts"][str(value)]["value"]
                    for key, value in required_data.items()
                }
                batch_data["name"] = molecule["name"]
                batch_data["smiles"] = smiles
                ic50_data.append(batch_data)
            except KeyError:
                # If any data is missing skip this molecule
                continue

        return ic50_data
