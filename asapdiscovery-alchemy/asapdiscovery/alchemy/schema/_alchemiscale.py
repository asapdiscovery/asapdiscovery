"""
:mod:`alchemiscale.models` --- common data models
=================================================

"""

from typing import Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from gufe.tokenization import GufeKey
from re import fullmatch
import unicodedata
import string


class Scope(BaseModel):
    org: Optional[str] = None
    campaign: Optional[str] = None
    project: Optional[str] = None

    def __init__(self, org=None, campaign=None, project=None):
        # we add this to allow for arg-based creation, not just keyword-based
        super().__init__(org=org, campaign=campaign, project=project)

    def __str__(self):
        triple = (
            i if i is not None else "*" for i in (self.org, self.campaign, self.project)
        )
        return "-".join(triple)

    def __lt__(self, other):
        return str(self) < str(other)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return str(self) == str(other)

    class Config:
        frozen = True

    @staticmethod
    def _validate_component(v, component):
        """
        Use regex to check that the component:
        - contains only alphanumeric or underscore characters, or is just a single asterisk
        - does not start with an underscore or number
        """

        # we require that there is a full match, so that the string is not
        # allowed to contain any other characters.
        if v is not None and not fullmatch(r"^[a-zA-Z][a-zA-Z0-9_]*|\*$", v):
            raise InvalidScopeError(
                f"'{component}' must either start with an alphabetical and contain "
                "only alphanumeric or underscore ('_') thereafter "
                "OR must be a single asterisk ('*')"
            )

        if v == "*":
            # if we're given an asterisk, cast this to `None` instead for
            # consistency
            v = None

        return v

    @validator("org")
    def valid_org(cls, v):
        return cls._validate_component(v, "org")

    @validator("campaign")
    def valid_campaign(cls, v):
        return cls._validate_component(v, "campaign")

    @validator("project")
    def valid_project(cls, v):
        return cls._validate_component(v, "project")

    @root_validator
    def check_scope_hierarchy(cls, values):
        if not _hierarchy_valid(values):
            raise InvalidScopeError(
                f"Invalid scope hierarchy: {values}, cannot specify wildcard ('*')"
                " in a scope component if a less specific scope component is not"
                " given, unless all components are wildcards (*-*-*)."
            )
        return values

    def to_tuple(self):
        return (self.org, self.campaign, self.project)

    @classmethod
    def from_str(cls, string):
        org, campaign, project = (i if i != "*" else None for i in string.split("-"))
        return cls(org=org, campaign=campaign, project=project)

    def is_superset(self, other: "Scope") -> bool:
        """Return `True` if this Scope is a superset of another.

        Check for a superset (not a proper superset) so that two equal scopes
        also return `True`.

        """
        if self.org is not None and self.org != other.org:
            return False
        if self.campaign is not None and self.campaign != other.campaign:
            return False
        if self.project is not None and self.project != other.project:
            return False
        return True

    def __repr__(self):  # pragma: no cover
        return f"<Scope('{str(self)}')>"

    def specific(self) -> bool:
        """Return `True` if this Scope has no unspecified elements."""
        return all(self.to_tuple())


class InvalidGufeKeyError(ValueError): ...


class ScopedKey(BaseModel):
    """Unique identifier for GufeTokenizables in state store.

    For this object, `org`, `campaign`, and `project` cannot contain wildcards.
    In other words, the Scope of a ScopedKey must be *specific*.

    """

    gufe_key: GufeKey
    org: str
    campaign: str
    project: str

    class Config:
        frozen = True

    @validator("gufe_key")
    def gufe_key_validator(cls, v):
        v = str(v)

        # GufeKey is of form <prefix>-<hex>
        try:
            _prefix, _token = v.split("-")
        except ValueError:
            raise InvalidGufeKeyError("gufe_key must be of the form '<prefix>-<hex>'")

        # Normalize the input to NFC form
        v_normalized = unicodedata.normalize("NFC", v)

        # Allowed characters: letters, numbers, underscores, hyphens
        allowed_chars = set(string.ascii_letters + string.digits + "_-")

        if not set(v_normalized).issubset(allowed_chars):
            raise InvalidGufeKeyError("gufe_key contains invalid characters")

        # Cast to GufeKey
        return GufeKey(v_normalized)

    def __repr__(self):  # pragma: no cover
        return f"<ScopedKey('{str(self)}')>"

    def __str__(self):
        return "-".join([self.gufe_key, self.org, self.campaign, self.project])

    def __lt__(self, other):
        return str(self) < str(other)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return str(self) == str(other)

    @classmethod
    def from_str(cls, string):
        try:
            prefix, token, org, campaign, project = string.split("-")
        except ValueError:
            raise ValueError("input does not appear to be a `ScopedKey`")

        gufe_key = GufeKey(f"{prefix}-{token}")

        return cls(gufe_key=gufe_key, org=org, campaign=campaign, project=project)

    @property
    def scope(self):
        return Scope(org=self.org, campaign=self.campaign, project=self.project)

    @property
    def qualname(self):
        return self.gufe_key.split("-")[0]

    def to_dict(self):
        return self.dict()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class InvalidScopeError(ValueError): ...


def _is_wildcard(char: Union[str, None]) -> bool:
    return char is None


def _find_wildcard(scope_list: list) -> Union[int, None]:
    """Finds the index of the first wildcard in a scope list."""
    for i, scope in enumerate(scope_list):
        if _is_wildcard(scope):
            return i
    return None


def _hierarchy_valid(scope_dict: dict[str : Union[str, None]]) -> bool:
    """Checks that the scope hierarchy is valid from a dictionary of scope components."""

    org = scope_dict.get("org")
    campaign = scope_dict.get("campaign")
    project = scope_dict.get("project")
    scope_list = [org, campaign, project]

    first_wildcard_ix = _find_wildcard(scope_list)
    if first_wildcard_ix is None:  # no wildcards, so we're good
        return True

    sublevels = scope_list[first_wildcard_ix:]
    # now check if any of the sublevels are not wildcards
    if any([not _is_wildcard(i) for i in sublevels]):
        return False
    return True
