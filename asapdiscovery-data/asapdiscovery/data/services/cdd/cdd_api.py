import json
import time
from typing import Optional

import pandas
from asapdiscovery.data.services.services_config import CDDSettings
from asapdiscovery.data.services.web_utils import _BaseWebAPI


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

    def get_molecules(
        self,
        smiles: Optional[str] = None,
        names: Optional[list[str]] = None,
        compound_ids: Optional[list[int]] = None,
    ) -> Optional[list[dict]]:
        """
        Search for molecules in the CDD vault.

        Notes:
            CDD only allows for a single structure searches via smiles, multiple molecules can be downloaded when using
            names or compound_ids.
            If molecule ids are missing in CDD we only return the subset that can be found

        Args:
            smiles: The smiles of the molecule to search for.
            names: The list of names of molecules which should be searched in the CDD.
            compound_ids: The list of CDD compound ids of molecules we wish to search for.

        Returns:
            A list of molecules found in the CDD.

        """
        if len([i for i in [smiles, names, compound_ids] if i is not None]) > 1:
            raise ValueError(
                "The arguments `smiles`, `names` and `compound_ids` are mutually exclusive provide only one."
            )

        mol_data = {"only_batch_ids": "true"}
        if smiles is not None:
            mol_data["structure"] = smiles
            mol_data["no_structures"] = "true"
            mol_data["structure_search_type"] = "exact"
        elif names is not None:
            mol_data["names"] = names
            mol_data["async"] = "true"
        else:
            mol_data["molecules"] = compound_ids
            mol_data["async"] = "true"
        result = json.loads(
            self._session.get(
                url=self.api_url + "molecules/", json=mol_data
            ).content.decode()
        )
        # handle missing molecules, originally found when searching moonshot data
        if "error" in result:
            import re

            # extract the list of missing molecule ids
            missing_mols = []
            for match in re.finditer("[0-9]+", result["error"]):
                missing_mols.append(int(match.group()))
            to_find = [mol for mol in compound_ids if mol not in missing_mols]
            mol_data["molecules"] = to_find
            # run the search again
            result = json.loads(
                self._session.get(
                    url=self.api_url + "molecules/", json=mol_data
                ).content.decode()
            )
        if "async" in mol_data:
            result = self.get_async_export(job_id=result["id"])
        if result["count"] == 0:
            return None
        else:
            return result["objects"]

    def get_protocols(
        self,
        protocol_names: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search for a specific protocol.

        Args:
            protocol_names: The list of protocol names to search for, if not provided all protocols will be pulled.

        Returns:
            A list of protocols associated with the given name
        """
        protocol_data = {}
        if protocol_names is not None:
            protocol_data["names"] = protocol_names
        result = self._session.get(url=self.api_url + "protocols", json=protocol_data)
        result_data = json.loads(result.content.decode())
        return result_data["objects"]

    def get_readout_rows(
        self,
        protocol: int,
        molecule_ids: Optional[list[int]] = None,
        types: Optional[list[str]] = None,
    ) -> Optional[list[dict]]:
        """
        Get the readout data for a specific protocol performed on a set of molecules.

        Args:
            molecule_ids: The CDD ids of the molecules to get the values for if None all molecules under this protocol will be downloaded.
            protocol: The id of the protocol to use in the search.
            types: A list of readout types to pull the results for.

        Returns:
            A dictionary of the readout data matching the search. The actual values are stored under `readouts`.
        """
        readout_data = {
            "protocols": [protocol],
            "async": "true",  # use async as we may have many results
        }
        if types is not None:
            readout_data["type"] = types
        if molecule_ids is not None:
            readout_data["molecules"] = molecule_ids
        result = self._session.get(url=self.api_url + "readout_rows", json=readout_data)
        request_id = json.loads(result.content.decode())["id"]
        result_data = self.get_async_export(job_id=request_id)
        if result_data["count"] == 0:
            return None
        else:
            return result_data["objects"]

    def get_async_export(self, job_id: int) -> dict:
        """
        A helper method to gather async request results.

        Args:
            job_id: The id of the request we want the results for.

        Notes:
            This function waits till the request is complete before returning the results.

        Returns:
            The finished request.
        """
        done = False
        while not done:
            result = json.loads(
                self._session.get(
                    url=self.api_url + f"exports/{job_id}"
                ).content.decode()
            )
            if "objects" not in result:
                time.sleep(1)
            else:
                return result

    def get_ic50_data(
        self, protocol_name: str
    ) -> Optional[
        pandas.DataFrame
    ]:  # TODO: remove duplication with the below readout method
        """
        A convenience method which wraps the required function calls to gather the raw ic50 data from the CDD for the
        calculated as part of the named protocol.

        Args:
            protocol_name: The name of the protocol we want all IC50 result for.

        Returns:
            A list of dictionaries containing the IC50 values along with upper and lower CI and curve class for each
            batch measurement on the molecules performed as part of the given protocol.

        """
        # get the id of the protocol we want the readout for
        protocols = self.get_protocols(protocol_names=[protocol_name])
        if protocols:
            protocol = protocols[0]
        else:
            return None
        # define the readouts we want to find and get the ids
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
        readout_data = self.get_readout_rows(
            protocol=protocol["id"], types=["batch_run_aggregate_row"]
        )
        # make a list of molecules we want to pull from the CDD
        compound_ids = set()
        # extract the results linking the molecules to the extracted data
        ic50_data = []
        for readout in readout_data:
            try:
                batch_data = {
                    f"{protocol_name}: {key}{' (µM)' if 'IC50' in key else ''}": readout[
                        "readouts"
                    ][
                        str(value)
                    ][
                        "value"
                    ]
                    for key, value in required_data.items()
                }
                # add a placeholder for the molecule data to be added later
                batch_data["name"] = readout["molecule"]
                batch_data["modified_at"] = readout["modified_at"]
                compound_ids.add(readout["molecule"])
                ic50_data.append(batch_data)
            except KeyError:
                # This is triggered if the upper and lower CI values are missing
                # This means the values falls outside the does series
                continue
        # gather the molecules
        molecule_data = self.get_molecules(compound_ids=list(compound_ids))
        compounds_by_id = {molecule["id"]: molecule for molecule in molecule_data}
        # loop over the list again and update the molecule info
        final_data = []
        for compound_data in ic50_data:
            try:
                mol_data = compounds_by_id[compound_data["name"]]
                compound_data["Smiles"] = mol_data["smiles"]
                compound_data["Inchi"] = mol_data["inchi"]
                compound_data["Inchi Key"] = mol_data["inchi_key"]
                compound_data["Molecule Name"] = mol_data["name"]
                compound_data["CXSmiles"] = mol_data["cxsmiles"]

                final_data.append(compound_data)
            except KeyError:
                continue

        return pandas.DataFrame(final_data)

    def get_readout(
        self, protocol_name: str, readout: str
    ) -> Optional[pandas.DataFrame]:
        # get the id of the protocol we want the readout for
        protocols = self.get_protocols(protocol_names=[protocol_name])
        if protocols:
            protocol = protocols[0]
        else:
            return None

        readout_ids = {}
        for readout_def in protocol["readout_definitions"]:
            readout_ids[readout_def["name"]] = readout_def["id"]

        if readout not in readout_ids:
            raise ValueError(
                f"Column {readout} not found in protocol {protocol_name}, available columns: {set(readout_ids.keys())}"
            )

        readout_data = self.get_readout_rows(protocol=protocol["id"])
        compound_ids = set()

        coldata = []
        for readout_elem in readout_data:
            try:
                batch_data = {}
                batch_data[readout] = readout_elem["readouts"][
                    str(readout_ids[readout])
                ]["value"]
                batch_data["name"] = readout_elem["molecule"]
                batch_data["modified_at"] = readout_elem["modified_at"]
                compound_ids.add(readout_elem["molecule"])
                coldata.append(batch_data)
            except KeyError:
                continue

        molecule_data = self.get_molecules(compound_ids=list(compound_ids))
        compounds_by_id = {molecule["id"]: molecule for molecule in molecule_data}
        final_data = []
        for compound_data in coldata:
            try:
                mol_data = compounds_by_id[compound_data["name"]]
                compound_data["Smiles"] = mol_data["smiles"]
                compound_data["Inchi"] = mol_data["inchi"]
                compound_data["Inchi Key"] = mol_data["inchi_key"]
                compound_data["Molecule Name"] = mol_data["name"]
                compound_data["CXSmiles"] = mol_data["cxsmiles"]

                final_data.append(compound_data)
            except KeyError:
                continue

        return pandas.DataFrame(final_data)
