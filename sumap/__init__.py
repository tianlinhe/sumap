from .sumap_ import SUMAP
from .sumap_ import SUMAP_nestedCV

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("sumap").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.0.0"


__all__ = ['SUMAP', 'SUMAP_nestedCV']
