from itertools import product
from typing import Literal

from asapdiscovery.data.selectors.ligand_selector import LigandSelectorBase
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.schema_v2.complex import Complex


class PairwiseLigandSelector(LigandSelectorBase):
    expander_type: Literal["PairwiseLigandSelector"] = "PairwiseLigandSelector"

    def _select(self, ligands: list[Ligand], complexes: list[Complex]):
        return list(product(ligands, complexes))

    def provenance(self):
        return {"selector": self.dict()}
