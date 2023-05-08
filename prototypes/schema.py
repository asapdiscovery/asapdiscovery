from enum import Enum
from pydantic import BaseModel, Field




class Ligand(BaseModel):
    id: Optional[str]
    vc_id_postera: Optional[str]
    moonshot_compound_id: Optional[str]
    target_id: Optional[str]
    source = str
    smiles: str

    ligand_provenance: Optional[ProvenanceBase]

    def to_sdf():
        ...

    def to_smiles():
        ...

    def to_oemol():
        ...

    @staticmethod
    def from_smiles():

    @staticmethod
    def from_sdf():

    @staticmethod
    def from_design_unit():

    @staticmethod
    def from_pdb():


class Target(BaseModel):
    id: str
    pdb_code: Optional[str]
    chain: str
    source: str
    reference_ligand: Optional[Ligand]
    protein_provenance: Optional[ProvenanceBase]

    def to_pdb():
        ...

    def to_design_unit():
        ...

    def to_oemol():
        ...

    def get_ligand(self):
        if not self.reference_ligand:
            raise ValueError("Target does not have a reference ligand")
        return self.reference_ligand

    @staticmethod
    def from_pdb():
        ...

    @staticmethod
    def from_design_unit():
        ...

    @staticmethod
    def from_oemol():
        ...



class ProvenanceBase():


class FragalysisProvenance(ProvenanceBase)

class RcsbProvenance(ProvenanceBase)



class PrepProvenance:
    seqres:
    loop_db:



class PreppedReceptor():
    protein: Protein
    reference_ligand: Optional[ReferenceLigand]
    prep_provenance: PrepProvenance


class DockingProvenance():
    method: str
    omega: str
    hybrid: str
    relax: str


class DockingInput(BaseModel):
    run_id: uuid.UUID4()
    run_name: str
    prepped_receptor: List[PreppedReceptor]
    query_ligands: List[QueryLigand]
    docking_provenance: DockingProvenance


class DockingOutput(BaseModel):
    input: DockingInput
    results: List[]


class DockingResults:
    chemgauss4:
    rmsd:
    posit:
    GAT:
    clash:
    schnet:
