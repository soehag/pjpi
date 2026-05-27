##################################################
# This is the init-file for the package 'regularisation'.
# Author: Hagen Söding
# Address: hagen.soeding@eaps.ethz.ch
# Date: 2024/08/05
##################################################


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