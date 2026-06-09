##################################################
# This is the init-file for the package 'gncore'.
# Author: Hagen Söding
# Address: hagen.soeding@eaps.ethz.ch
# Date: 2024/08/05
##################################################


# List of all modules in the package
__all__ = ['gaussnewtoncore', 'geophysical', 'petrophysical']

from . import gaussnewtoncore
from . import geophysical
from . import petrophysical
from . import physicsanddata