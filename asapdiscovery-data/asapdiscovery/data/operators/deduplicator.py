from asapdiscovery.data.schema.ligand import Ligand
from pydantic.v1 import BaseModel


class LigandDeDuplicator(BaseModel):
    """
    Class to deduplicate ligands based on their inchikey
    """

    def deduplicate(self, ligands: list[Ligand]):
        seen_values = set()
        deduplicated_list = []
        for lig in ligands:
            inchikey = lig.inchikey
            if inchikey not in seen_values:
                seen_values.add(inchikey)
                deduplicated_list.append(lig)

        return deduplicated_list
