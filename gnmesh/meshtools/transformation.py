"""
Transformations of the form y = f(x) together with their inverses and derivatives.

The classes in this module are intentionally small and explicit so they can be used
directly in demos and optimisation code without a larger abstraction layer.

Available classes:
- MultiplicativeTransformation: Scale values by a constant factor.
- PowerTransformation: Raise values to a fixed power.
- InverseTransformation: Identity transformation used where an inverse object is needed.
- LogarithmicBarrierTransformationGreaterThan: Map x > barrier to the real line.
- LogarithmicBarrierTransformationLessThan: Map x < barrier to the real line.
- LogarithmicBarrierTransformationTwoSided: Map lower_barrier < x < upper_barrier to the real line.

No module-level functions are defined.

Author: Hagen Söding
Affiliation: ETH Zürich
Email: hagen.soeding@eaps.ethz.ch
"""


import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultiplicativeTransformation:
    """Scale values by a constant multiplier."""

    def __init__(self, multiplier):
        assert multiplier != 0, "Multiplier must not be zero."
        self._multiplier = multiplier

    def forward(self, x):
        """ Multiplicative transformation. """
        return x * self._multiplier

    def backward(self, x):
        """ Inverse of the multiplicative transformation. """
        return x / self._multiplier
    
    def derivative_forward(self, x):
        """ Derivative of the multiplicative transformation. """
        return self._multiplier * np.ones_like(x)

    def derivative_backward(self, x):
        """ Derivative of the inverse of the multiplicative transformation. """
        return 1 / self._multiplier * np.ones_like(x)

class PowerTransformation:
    """Raise values to a fixed power and provide the matching inverse."""

    def __init__(self, power):
        assert power != 0, "Power must not be zero."
        self._power = power

    def forward(self, x):
        """ Power transformation. """
        return x ** self._power

    def backward(self, x):
        """ Inverse of the power transformation. """
        return x ** (1 / self._power)
    
    def derivative_forward(self, x):
        """ Derivative of the power transformation. """
        return self._power * x ** (self._power - 1)

    def derivative_backward(self, x):
        """ Derivative of the inverse of the power transformation. """
        return 1 / self._power * x ** (1 / self._power - 1)

# Identity transform used as a lightweight inverse placeholder.
InverseTransformation = PowerTransformation(1)

class LogarithmicBarrierTransformationGreaterThan:
    """Map values above a lower barrier to the real line via a logarithm."""

    def __init__(self, barrier, eps=1e-6):
        self._barrier = barrier
        self._eps = eps

    def forward(self, x):
        """ Logarithmic barrier transformation for x > barrier. """
        return np.log(x - self._barrier)

    def backward(self, x):
        """ Inverse of the logarithmic barrier transformation for x > barrier. """
        unclipped_backward = self._barrier + np.exp(x)
        # Clip only when the inverse gets numerically too close to the barrier.
        if np.any(unclipped_backward < self._barrier + self._eps):
            import logging
            logging.getLogger(__name__).warning("Warning: Inverse of logarithmic barrier transformation is clipped.")
        return np.clip(unclipped_backward, self._barrier + self._eps, None)
    
    def derivative_forward(self, x):
        """ Derivative of the logarithmic barrier transformation for x > barrier. Is analytical equal to 1/(x - barrier). """
        sign = np.sign(x - self._barrier)
        absolute = np.abs(x - self._barrier)
        return sign / (absolute + self._eps)

    def derivative_backward(self, x):
        """ Derivative of the inverse of the logarithmic barrier transformation for x > barrier. Is analytical equal to exp(x). """
        return np.exp(x)

class LogarithmicBarrierTransformationLessThan:
    """Map values below an upper barrier to the real line via a logarithm."""

    def __init__(self, barrier, eps=1e-6):
        self._barrier = barrier
        self._eps = eps

    def forward(self, x):
        """Logarithmic barrier transformation for x < barrier. 
        The minus sign is added to make the function go to -inf close to the boundary."""
        return -np.log(self._barrier - x)

    def backward(self, x):
        """ Inverse of the logarithmic barrier transformation for x < barrier. """
        unclipped_backward = self._barrier - np.exp(-x)
        if np.any(unclipped_backward > self._barrier - self._eps):
            import logging
            logging.getLogger(__name__).warning("Warning: Inverse of logarithmic barrier transformation is clipped.")
        return np.clip(unclipped_backward, None, self._barrier - self._eps)
    
    def derivative_forward(self, x):
        """ Derivative of the logarithmic barrier transformation for x < barrier. Is analytical equal to 1/(barrier - x). """
        sign = np.sign(self._barrier - x)
        absolute = np.abs(self._barrier - x)
        return sign / (absolute + self._eps)

    def derivative_backward(self, x):
        """ Derivative of the inverse of the logarithmic barrier transformation for x < barrier. Is analytical equal to exp(-x). """
        return np.exp(-x)

class LogarithmicBarrierTransformationTwoSided:
    """Map an open interval to the real line with logarithmic barriers at both ends."""

    def __init__(self, lower_barrier, upper_barrier, eps_barrier=1e-6, eps_exp=30, derivative="numeric"):
        self._lower_barrier = lower_barrier
        self._upper_barrier = upper_barrier
        assert lower_barrier < upper_barrier, "Lower barrier must be smaller than upper barrier."
        self._eps_barrier = eps_barrier
        self._eps_exp = eps_exp
        assert derivative in ["analytic", "numeric"], "Derivative must be either 'analytic' or 'numeric'."
        self._derivative = derivative

    def forward(self, x):
        """ Logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        return np.log(x - self._lower_barrier) - np.log(self._upper_barrier - x)

    def backward(self, x):
        """ Inverse of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        if np.any(np.abs(x) > self._eps_exp):
            # Clip x to avoid overflow in np.exp(x).
            logger.warning("Warning: Inverse of logarithmic barrier transformation is clipped")

        x_clipped = np.clip(x, a_min=-self._eps_exp, a_max=self._eps_exp)

        # Clip the result to keep it strictly inside the admissible interval and avoid boundary issues.
        unclipped_backward = (self._lower_barrier + self._upper_barrier * np.exp(x_clipped)) / (1 + np.exp(x_clipped))
        if np.any(unclipped_backward < self._lower_barrier + self._eps_barrier):
            logger.warning("Warning: Inverse of logarithmic barrier transformation is clipped from below.")
        if np.any(unclipped_backward > self._upper_barrier - self._eps_barrier):
            logger.warning("Warning: Inverse of logarithmic barrier transformation is clipped from above.")
        return np.clip(unclipped_backward, self._lower_barrier + self._eps_barrier, self._upper_barrier - self._eps_barrier)

        
    def derivative_forward(self, x):
        """ Derivative of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        # Keep the analytic/numeric switch local to the class so the public API stays simple.
        if self._derivative == "analytic":
            return self.derivative_forward_analytic(x)
        else:
            return self.derivative_forward_numeric(x)
    
    def derivative_backward(self, x):
        """ Derivative of the inverse of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        if self._derivative == "analytic":
            return self.derivative_backward_analytic(x)
        else:
            return self.derivative_backward_numeric(x)

    def derivative_forward_analytic(self, x):
        """ Derivative of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        return 1.0 / (x - self._lower_barrier) + 1.0 / (self._upper_barrier - x)
    
    def derivative_forward_numeric(self, x):
        """ Numerical derivative of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        # If x is closer to the lower barrier than the upper barrier, then add eps to x.
        # If x is closer to the upper barrier than the lower barrier, then subtract eps from x.
        # This function works vectorized and avoids stepping outside the interval.

        forward_difference_indices = np.abs(x - self._lower_barrier) < np.abs(self._upper_barrier - x)
        backward_difference_indices = ~forward_difference_indices

        forward_difference = np.zeros_like(x)
        backward_difference = np.zeros_like(x)

        forward_difference[forward_difference_indices] = (self.forward(x[forward_difference_indices] + self._eps_barrier) - self.forward(x[forward_difference_indices])) / self._eps_barrier
        backward_difference[backward_difference_indices] = (self.forward(x[backward_difference_indices]) - self.forward(x[backward_difference_indices] - self._eps_barrier)) / self._eps_barrier

        return forward_difference + backward_difference

    def derivative_backward_analytic(self, x):
        """ Derivative of the inverse of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        if np.any(np.abs(x) > self._eps_exp):
            # Clip x to avoid overflow in np.exp(x).
            logger.warning("Warning: Inverse of logarithmic barrier transformation is clipped")
            x_clipped = np.clip(x, a_min=-self._eps_exp, a_max=self._eps_exp)
        return (np.exp(x_clipped) * (self._upper_barrier - self._lower_barrier))/(1 + np.exp(x_clipped))**2
        
    def derivative_backward_numeric(self, x):
        """ Numerical derivative of the inverse of the logarithmic barrier transformation for lower_barrier < x < upper_barrier. """
        # If x is closer to the lower barrier than the upper barrier, then add eps to x.
        # If x is closer to the upper barrier than the lower barrier, then subtract eps from x.
        # This function works vectorized and avoids stepping outside the interval.

        forward_difference_indices = x < 0
        backward_difference_indices = ~forward_difference_indices

        forward_difference = np.zeros_like(x)
        backward_difference = np.zeros_like(x)

        forward_difference[forward_difference_indices] = (self.backward(x[forward_difference_indices] + self._eps_barrier) - self.backward(x[forward_difference_indices])) / self._eps_barrier
        backward_difference[backward_difference_indices] = (self.backward(x[backward_difference_indices]) - self.backward(x[backward_difference_indices] - self._eps_barrier)) / self._eps_barrier

        return forward_difference + backward_difference