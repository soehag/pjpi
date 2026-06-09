"""
gncore package initialization.

Public modules exported by this package:

- ``gaussnewtoncore``: shared core API for Gauss-Newton managers
- ``geophysical``: geophysical inversion manager
- ``petrophysical``: petrophysical inversion manager
- ``petrophysical_decoupled``: petrophysical inversion manager with decoupled updates
- ``physicsanddata``: helpers coupling methods and data
"""

__all__ = [
	'gaussnewtoncore',
	'geophysical',
	'petrophysical',
    'petrophysical_decoupled',
	'physicsanddata',
]

from . import gaussnewtoncore
from . import geophysical
from . import petrophysical
from . import petrophysical_decoupled
from . import physicsanddata