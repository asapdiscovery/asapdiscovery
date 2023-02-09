"""Computational chemistry packages developed by the ASAP Discovery Consortium."""
# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

from .mcs import *
from .docking import *
from .analysis import *
from .rocauc import *
from .modeling import *
