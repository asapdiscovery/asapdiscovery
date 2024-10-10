import abc
from typing import Literal

from pydantic.v1 import Field, PositiveFloat, PositiveInt

from .base import _SchemaBase


class _BaseAtomMapper(_SchemaBase):
    """
    A base atom mapper which should be used to configure the method used to generate atom mappings.
    """

    @abc.abstractmethod
    def _get_mapper(self): ...

    def get_mapper(self):
        return self._get_mapper()

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        """
        Gather the names and versions of the Software used to calculate the atom mapping.
        """
        ...


class LomapAtomMapper(_BaseAtomMapper):
    """
    A settings class for the LomapAtomMapper in openFE
    """

    type: Literal["LomapAtomMapper"] = "LomapAtomMapper"

    timeout: PositiveInt = Field(
        20, description="The timeout in seconds of the MCS algorithm pass to RDKit"
    )
    threed: bool = Field(
        True,
        description="If positional info should used to choose between symmetrically equivalent mappings and prune the mapping.",
    )
    max3d: PositiveFloat = Field(
        1000.0,
        description="Maximum discrepancy in Angstroms between atoms before the mapping is not allowed. Large numbers trim no atoms.",
    )
    element_change: bool = Field(
        True, description="Whether to allow element changes in the mappings."
    )
    seed: str = Field(
        "", description="The SMARTS string used as a seed for MCS searches."
    )
    shift: bool = Field(
        True,
        description="When determining 3D overlap, if to translate the two MCS to minimise the RMSD to boost potential alignment.",
    )

    def _get_mapper(self):
        from openfe import LomapAtomMapper

        # TODO use an alias once we can use pydantic-2
        data = self.dict(exclude={"type", "timeout"})
        data["time"] = self.timeout
        return LomapAtomMapper(**data)

    def provenance(self) -> dict[str, str]:
        import lomap
        import openfe
        import rdkit

        return {
            "openfe": openfe.__version__,
            "lomap": lomap.__version__,
            "rdkit": rdkit.__version__,
        }


class PersesAtomMapper(_BaseAtomMapper):
    """
    A settings class for the PersesAtomMapper in openFE
    """

    type: Literal["PersesAtomMapper"] = "PersesAtomMapper"

    allow_ring_breaking: bool = Field(
        True, description="If only full cycles of the molecules should be mapped."
    )
    preserve_chirality: bool = Field(
        True,
        description="If mappings must strictly preserve the chirality of the molecules.",
    )
    use_positions: bool = Field(
        True,
        description="If 3D positions should be used during the generation of the mappings.",
    )
    coordinate_tolerance: PositiveFloat = Field(
        0.25,
        description="A tolerance on how close coordinates need to be in Angstroms before they can be mapped. Does nothing if use_positions is `False`.",
    )

    def _get_mapper(self):
        from openfe import PersesAtomMapper

        return PersesAtomMapper(**self.dict(exclude={"type"}))

    def provenance(self) -> dict[str, str]:
        import openeye.oechem
        import openfe
        import perses

        return {
            "openfe": openfe.__version__,
            "perses": perses.__version__,
            "openeye.oechem": openeye.oechem.OEChemGetVersion(),
        }


class KartografAtomMapper(_BaseAtomMapper):
    """
    A settings class for the kartograf atom mapping method.
    """

    type: Literal["KartografAtomMapper"] = "KartografAtomMapper"

    map_exact_ring_matches_only: bool = Field(
        True, description="If only rings should be matched to other rings."
    )
    atom_max_distance: float = Field(
        0.95,
        description="The distance in Angstroms between two atoms before they can not be matched.",
    )
    atom_map_hydrogens: bool = Field(
        True, description="If hydrogens should also be mapped in the transform."
    )
    map_hydrogens_on_hydrogens_only: bool = Field(
        False, description="If hydrogens should only be matched to other hydrogens."
    )
    mapping_algorithm: Literal["linear_sum_assignment", "minimal_spanning_tree"] = (
        Field(
            "linear_sum_assignment",
            description="The mapping algorithm that should be used.",
        )
    )

    def _get_mapper(self):
        from kartograf.atom_mapper import KartografAtomMapper, mapping_algorithm

        # workaround the awkward argument name
        settings = self.dict(exclude={"type", "mapping_algorithm"})
        settings["_mapping_algorithm"] = (
            mapping_algorithm.linear_sum_assignment
            if self.mapping_algorithm == "linear_sum_assignment"
            else mapping_algorithm.minimal_spanning_tree
        )
        return KartografAtomMapper(**settings)

    def provenance(self) -> dict[str, str]:
        import kartograf._version
        import openfe
        import rdkit

        return {
            "openfe": openfe.__version__,
            "rdkit": rdkit.__version__,
            "kartograf": kartograf._version.__version__,
        }
