from asapdiscovery.data.schema_v2.ligand import Ligand
from pydantic import BaseModel


class LigandDeDuplicator(BaseModel):
    """
    Class to deduplicate ligands based on their inchikey
    """

    def deduplicate(self, ligands: list[Ligand]):
        seen_values = set()
        deduplicated_list = []
        for l in ligands:
            inchikey = l.inchikey
            if inchikey not in seen_values:
                seen_values.add(inchikey)
                deduplicated_list.append(l)

        return deduplicated_list
