##################################################
# This is the init-file for the package 'gnmesh'.
# Author: Hagen Söding
# Address: hagen.soeding@eaps.ethz.ch
# Date: 2024/08/05
##################################################


# List of all modules in the package
__all__ = ['gncore', 'regularisation', 'meshtools']


def __getattr__(name):
	if name in __all__:
		import importlib

		module = importlib.import_module(f".{name}", __name__)
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")