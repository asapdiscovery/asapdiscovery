import json
from collections import Counter
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from asapdiscovery.data.schema.ligand import Ligand


def check_ligand_series_uniqueness_and_names(ligands: list["Ligand"]) -> None:
    """
    Check the ligands are unique and have names.

    Args:
        ligands: The ligands to check for uniqueness and names.

    Raises:
        ValueError: If the ligands are not unique or have no names.
    """
    if len(set(ligands)) != len(ligands):
        count = Counter(ligands)
        duplicated = [key.compound_name for key, value in count.items() if value > 1]
        raise ValueError(
            f"ligand series contains {len(duplicated)} duplicate ligands: {duplicated}"
        )

    # if any ligands lack a name, then raise an exception; important for
    # ligands to have names for human-readable result gathering downstream
    if missing := len([ligand for ligand in ligands if not ligand.compound_name]):
        raise ValueError(
            f"{missing} of {len(ligands)} ligands do not have names; names are required for ligands for downstream results handling"
        )


#: The atom-mapping scorers ASAP-Alchemy exposes in its schema.
SCORERS = ("default_lomap", "default_perses")


def get_mapping_scorer(scorer_name: str):
    """Resolve an atom-mapping scorer name to the openfe callable that implements it.

    ``default_perses`` is kept in the schema so that networks planned on the openfe 1.8
    stack still deserialise, but openfe 1.12 (the ``alchemy-membrane`` stack) removed
    ``perses_scorers`` along with the perses dependency itself, so resolving it there
    fails with a clear message rather than an ImportError at module import time.

    Args:
        scorer_name: One of :data:`SCORERS`.

    Raises:
        ValueError: If ``scorer_name`` is not a known scorer.
        NotImplementedError: If the scorer is not available in the installed openfe.
    """
    import openfe

    if scorer_name == "default_lomap":
        return openfe.lomap_scorers.default_lomap_score

    if scorer_name == "default_perses":
        if not hasattr(openfe, "perses_scorers"):
            raise NotImplementedError(
                f"The 'default_perses' scorer is unavailable: openfe "
                f"{openfe.__version__} no longer ships perses_scorers. Use "
                "'default_lomap' instead, or plan on the openfe 1.8 ('alchemy') "
                "environment."
            )
        return openfe.perses_scorers.default_perses_scorer

    raise ValueError(
        f"Atom mapping scorer {scorer_name} not recognized; use one of {list(SCORERS)}."
    )


def is_presolvated_receptor(component) -> bool:
    """True if ``component`` is a receptor that carries its own solvent and box.

    A gufe ``SolvatedPDBComponent`` (and its ``ProteinMembraneComponent`` subclass) is
    supplied as an already-solvated, already-equilibrated PDB with explicit box vectors,
    so its waters, ions and box come from the file. Such a system must **not** also be
    given a ``SolventComponent``.

    Returns False on openfe <1.11, where the class does not exist.
    """
    import gufe

    solvated_cls = getattr(gufe, "SolvatedPDBComponent", None)
    return solvated_cls is not None and isinstance(component, solvated_cls)


def is_membrane_receptor(component) -> bool:
    """True if ``component`` is a membrane-embedded receptor.

    A gufe ``ProteinMembraneComponent`` is the marker type that tells the protocol to use
    a ``MonteCarloMembraneBarostat`` (anisotropic along z) rather than an isotropic one.
    Every membrane receptor is also a pre-solvated one; the converse is not true.

    Returns False on openfe <1.11, where the class does not exist.
    """
    import gufe

    membrane_cls = getattr(gufe, "ProteinMembraneComponent", None)
    return membrane_cls is not None and isinstance(component, membrane_cls)


#: gufe component classes whose receptors carry their own solvent and box.
_PRESOLVATED_QUALNAMES = frozenset({"SolvatedPDBComponent", "ProteinMembraneComponent"})
#: ...and the subset of those that are membrane-embedded.
_MEMBRANE_QUALNAMES = frozenset({"ProteinMembraneComponent"})


def _serialised_qualname(receptor_json: str) -> Optional[str]:
    """The gufe class name recorded in a serialised component, without rebuilding it.

    ``GufeTokenizable.to_json`` emits a keyed chain -- a list of ``[key, payload]`` pairs
    whose last entry is the top-level object -- and every payload carries the class in
    ``__qualname__``. Reading that is O(size of JSON); actually deserialising a membrane
    receptor rebuilds a ~150,000-atom RDKit molecule, which is far too expensive for a
    type check.

    Returns None if the payload is not in the expected shape, so callers fall back to
    real deserialisation rather than guessing wrong.
    """
    try:
        payload = json.loads(receptor_json)
    except (TypeError, ValueError):
        return None
    # keyed chain: [[key, {...}], ...]; plain dict form: {...}
    if isinstance(payload, list) and payload:
        last = payload[-1]
        if isinstance(last, (list, tuple)) and len(last) == 2:
            payload = last[1]
    if isinstance(payload, dict):
        qualname = payload.get("__qualname__")
        if isinstance(qualname, str):
            return qualname
    return None


def receptor_json_is_presolvated(receptor_json: str) -> Optional[bool]:
    """Whether a serialised receptor carries its own solvent, or None if undeterminable."""
    qualname = _serialised_qualname(receptor_json)
    return None if qualname is None else qualname in _PRESOLVATED_QUALNAMES


def receptor_json_is_membrane(receptor_json: str) -> Optional[bool]:
    """Whether a serialised receptor is membrane-embedded, or None if undeterminable."""
    qualname = _serialised_qualname(receptor_json)
    return None if qualname is None else qualname in _MEMBRANE_QUALNAMES
