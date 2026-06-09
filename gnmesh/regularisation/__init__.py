"""
regularisation package exports.

This package contains a collection of regularisation operators and
helpers used by the inversion managers. It exposes damping, smoothing,
gradient- and anisotropic-regularisation implementations as well as a
small linear-operator wrapper used to compose regularisation matrices.
"""

from .damping import DampingReferenceModel, DampingStepWidth
from .smoothingpg import FirstOrderSmoothing, SecondOrderSmoothing
from .xgradient import XGradient, XGradientReferenceModel, XGradientSingleModel
from .linearoperator import LinearOperator
from .jointtotalvariation import JointTotalVariation
from .smoothingspatial import FirstOrderSmoothingGradient, FirstOrderSmoothingSquaredGradient, LaplacianSmoothing
from .anisotropicsmoothing import aniso_homogeneous_kernel, aniso_heterogeneous_kernel, anisokernel_jacobian, AnisotropicSmoothing, plot_all_kernels, plot_aniso_diffusion_tensor_field, plot_heterogeneous_kernel, plot_homogeneous_kernel, plot_kernel

__all__ = [
	"DampingReferenceModel",
	"DampingStepWidth",
	"FirstOrderSmoothing",
	"SecondOrderSmoothing",
	"XGradient",
	"XGradientReferenceModel",
	"XGradientSingleModel",
	"LinearOperator",
	"JointTotalVariation",
	"FirstOrderSmoothingGradient",
	"FirstOrderSmoothingSquaredGradient",
	"LaplacianSmoothing",
	"aniso_homogeneous_kernel",
	"aniso_heterogeneous_kernel",
	"anisokernel_jacobian",
	"AnisotropicSmoothing",
	"plot_all_kernels",
	"plot_aniso_diffusion_tensor_field",
	"plot_heterogeneous_kernel",
	"plot_homogeneous_kernel",
	"plot_kernel",
]