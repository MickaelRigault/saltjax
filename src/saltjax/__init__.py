"""Fast batched SALT2 / SALT3 lightcurve fitting with JAX, reproducing sncosmo."""

__version__ = "0.2.0"

from .fit import fit_salt # noqa: F401, E402
from .tables import build_tables, get_tables, get_salt_source # noqa: F401, E402
from .data import pack_lightcurves, band_integrals # noqa: F401, E402
from .effects import get_effects # noqa: F401, E402
from .model import get_model_functions # noqa: F401, E402
from .kernel import kernel_weights # noqa: F401, E402
