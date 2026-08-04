import re
import warnings

# Patch warnings.showwarning instead of using filterwarnings.
#
# Third-party packages (notably pandas.core.nanops at module-level) call
# warnings.simplefilter("always", DeprecationWarning) many times during import,
# which repeatedly pushes an "always" entry to the front of the filter list and
# overrides any "ignore" filters we add.  Patching showwarning is evaluated after
# all filter logic and cannot be overridden by third-party filter manipulation.
_SUPPRESSED_DEPRECATION_PATTERNS = re.compile(
    r"builtin type \S+ has no __module__ attribute"
    r"|Support for class-based `config` is deprecated"
    r"|Partial charges have been provided, these will preferentially be used"
)
_orig_showwarning = warnings.showwarning


def _showwarning(msg, cat, fname, lno, *args, **kwargs):
    if issubclass(cat, (DeprecationWarning, UserWarning)) and _SUPPRESSED_DEPRECATION_PATTERNS.search(str(msg)):
        return
    _orig_showwarning(msg, cat, fname, lno, *args, **kwargs)


warnings.showwarning = _showwarning

from asapdiscovery.alchemy.cli.alchemy import alchemy
from asapdiscovery.alchemy.cli.bespoke import bespoke
from asapdiscovery.alchemy.cli.prep import prep

alchemy.add_command(prep)
alchemy.add_command(bespoke)
