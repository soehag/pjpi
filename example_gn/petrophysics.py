import numpy as np


K_CO2 = 0.01  # Gpa
RHO_CO2 = 231.53  # kg/m^3

K_BRINE = 3.63  # Gpa
RHO_BRINE = 1164.59  # kg/m^3

K_MATRIX = 37.78  # Gpa
RHO_MATRIX = 2670.89  # kg/m^3

K_AIR = 1.42 * 10**5 * 10**-9  # Gpa
K_DRY = 3.5  # Gpa for sample 1

S_WAVE_VELOCITY_IVANOVA = 1.42
RHO_IVANOVA = 4050
PHI = 0.28

SATURATION_EXPONENT = 1.62
CEMENTATION_EXPONENT = 2.0
R_FLUID = 3e-2
MINIMUM_C02_SATURATION = 0.0
MAXIMUM_C02_SATURATION = 0.55


class Transformation:
    """Base class for paired forward and inverse transformations.

    The class stores a forward callable, its inverse, and a finite-difference
    strategy for derivative evaluation. Subclasses provide the physical
    transformation logic and connect the base class to their own methods.

    If analytic derivative expressions are available, pass them via
    ``derivative_function_analytic`` and/or ``derivative_inverse_analytic``.
    In that case the corresponding finite-difference setting is ignored and the
    derivative mode switches to ``"analytic"`` automatically.
    """

    def __init__(
        self,
        forward,
        inverse,
        derivative_function_analytic=None,
        derivative_inverse_analytic=None,
        eps=1e-4,
        difference_forward="forward",
        difference_inverse="forward",
    ):
        self._function = forward
        self._inverse_function = inverse
        self._eps = eps

        if derivative_function_analytic is not None:
            print("Using analytic derivative for the forward transformation.")
            self._difference_function = "analytic"
        else:
            print("Using finite-difference derivative for the forward transformation.")
            assert difference_forward in ["forward", "backward"], "Difference must be either forward or backward"
            self._difference_function = difference_forward

        if derivative_inverse_analytic is not None:
            print("Using analytic derivative for the inverse transformation.")
            self._difference_inverse = "analytic"
        else:
            print("Using finite-difference derivative for the inverse transformation.")
            assert difference_inverse in ["forward", "backward"], "Difference must be either forward or backward"
            self._difference_inverse = difference_inverse

        self._derivative_function_analytic = derivative_function_analytic
        self._derivative_inverse_analytic = derivative_inverse_analytic

    def forward(self, x):
        """Apply the forward transformation."""
        return self._function(x)

    def backward(self, x):
        """Apply the inverse transformation."""
        return self._inverse_function(x)

    def derivative_function(self, x):
        """Return the derivative of the forward transformation."""
        if self._difference_function == "forward":
            differential = (self._function(x + self._eps) - self._function(x)) / self._eps
        elif self._difference_function == "backward":
            differential = (self._function(x) - self._function(x - self._eps)) / self._eps
        elif self._difference_function == "analytic":
            assert self._derivative_function_analytic is not None
            differential = self._derivative_function_analytic(x)
        else:
            raise ValueError("Difference must be either forward or backward")
        return differential

    def derivative_inverse(self, y):
        """Return the derivative of the inverse transformation."""
        if self._difference_inverse == "forward":
            differential = (self._inverse_function(y + self._eps) - self._inverse_function(y)) / self._eps
        elif self._difference_inverse == "backward":
            differential = (self._inverse_function(y) - self._inverse_function(y - self._eps)) / self._eps
        elif self._difference_inverse == "analytic":
            assert self._derivative_inverse_analytic is not None
            differential = self._derivative_inverse_analytic(y)
        else:
            raise ValueError("Difference must be either forward or backward")
        return differential

    def derivative_forward(self, x):
        """Convenience alias for the forward derivative."""
        return self.derivative_function(x)

    def derivative_backward(self, y):
        """Convenience alias for the inverse derivative."""
        return self.derivative_inverse(y)


class GassmannTransformation(Transformation):
    """P-wave velocity and saturation transformation based on Gassmann.

    Parameters
    ----------
    k_co2 : float
        Bulk modulus of the CO2 phase.
    rho_co2 : float
        Density of the CO2 phase.
    k_brine : float
        Bulk modulus of the brine phase.
    rho_brine : float
        Density of the brine phase.
    k_matrix : float
        Bulk modulus of the rock matrix.
    rho_matrix : float
        Density of the rock matrix.
    k_air : float
        Bulk modulus of air used in the Hill-type substitution.
    k_dry : float
        Dry-rock bulk modulus.
    s_wave_velocity : float
        S-wave velocity used in the shear-modulus proxy.
    rho_ivanova : float
        Density used in the shear-modulus proxy.
    phi : float
        Porosity used for the rock-fluid substitution.
    version_gassmann : str
        Gassmann variant to use, currently ``"ivanova"`` or ``"hs"``.
    version_fluid_mixtures : str
        Fluid mixture law, currently ``"wood"``, ``"domenico"`` or ``"brie"``.
    brie_exponent : float
        Exponent used by the Brie fluid-mixture law.
    correction_factor : float
        Multiplicative scaling applied to the resulting P-wave velocity.
    maximum_saturation : float
        Upper bound used when numerically inverting velocity to saturation.
    n_saturation : int
        Number of saturation samples used for numerical inversion.
    eps : float
        Finite-difference step size for numerical derivatives.
    derivative_function_analytic : callable | None
        Analytic derivative of the forward transformation. If provided, the
        forward derivative uses this callable instead of finite differences.
    derivative_inverse_analytic : callable | None
        Analytic derivative of the inverse transformation. If provided, the
        inverse derivative uses this callable instead of finite differences.
    difference_forward : str
        Finite-difference scheme for the forward derivative. Possible values
        are ``"forward"`` and ``"backward"``. Ignored when an analytic
        derivative is supplied.
    difference_inverse : str
        Finite-difference scheme for the inverse derivative. Possible values
        are ``"forward"`` and ``"backward"``. Ignored when an analytic
        derivative is supplied.
    """

    def __init__(
        self,
        k_co2=K_CO2,
        rho_co2=RHO_CO2,
        k_brine=K_BRINE,
        rho_brine=RHO_BRINE,
        k_matrix=K_MATRIX,
        rho_matrix=RHO_MATRIX,
        k_air=K_AIR,
        k_dry=K_DRY,
        s_wave_velocity=S_WAVE_VELOCITY_IVANOVA,
        rho_ivanova=RHO_IVANOVA,
        phi=PHI,
        version_gassmann="ivanova",
        version_fluid_mixtures="brie",
        brie_exponent=3,
        correction_factor=10**4.5,
        maximum_saturation=0.55,
        n_saturation=10000,
        eps=1e-4,
        derivative_function_analytic=None,
        derivative_inverse_analytic=None,
        difference_forward="forward",
        difference_inverse="forward",
    ):
        """Create a Gassmann transformation instance.

        Parameters
        ----------
        k_co2 : float
            Bulk modulus of the CO2 phase.
        rho_co2 : float
            Density of the CO2 phase.
        k_brine : float
            Bulk modulus of the brine phase.
        rho_brine : float
            Density of the brine phase.
        k_matrix : float
            Bulk modulus of the rock matrix.
        rho_matrix : float
            Density of the rock matrix.
        k_air : float
            Bulk modulus of air used in the Hill-type substitution.
        k_dry : float
            Dry-rock bulk modulus.
        s_wave_velocity : float
            S-wave velocity used in the shear-modulus proxy.
        rho_ivanova : float
            Density used in the shear-modulus proxy.
        phi : float
            Porosity used for the rock-fluid substitution.
        version_gassmann : str
            Gassmann variant to use, currently ``"ivanova"`` or ``"hs"``.
        version_fluid_mixtures : str
            Fluid mixture law, currently ``"wood"``, ``"domenico"`` or ``"brie"``.
        brie_exponent : float
            Exponent used by the Brie fluid-mixture law.
        correction_factor : float
            Multiplicative scaling applied to the resulting P-wave velocity.
        maximum_saturation : float
            Upper bound used when numerically inverting velocity to saturation.
        n_saturation : int
            Number of saturation samples used for numerical inversion.
        eps : float
            Finite-difference step size for numerical derivatives.
        derivative_function_analytic : callable | None
            Analytic derivative of the forward transformation. If provided,
            it replaces the finite-difference approximation.
        derivative_inverse_analytic : callable | None
            Analytic derivative of the inverse transformation. If provided,
            it replaces the finite-difference approximation.
        difference_forward : str
            Finite-difference scheme for the forward derivative. Possible
            values are ``"forward"`` and ``"backward"``.
        difference_inverse : str
            Finite-difference scheme for the inverse derivative. Possible
            values are ``"forward"`` and ``"backward"``.
        """
        self._k_co2 = k_co2
        self._rho_co2 = rho_co2
        self._k_brine = k_brine
        self._rho_brine = rho_brine
        self._k_matrix = k_matrix
        self._rho_matrix = rho_matrix
        self._k_air = k_air
        self._k_dry = k_dry
        self._s_wave_velocity = s_wave_velocity
        self._rho_ivanova = rho_ivanova
        self._phi = phi
        self._version_gassmann = version_gassmann
        self._version_fluid_mixtures = version_fluid_mixtures
        self._brie_exponent = brie_exponent
        self._correction_factor = correction_factor
        self._maximum_saturation = maximum_saturation
        self._n_saturation = n_saturation

        super().__init__(
            forward=self.saturation_to_vp,
            inverse=self.vp_to_saturation,
            derivative_function_analytic=derivative_function_analytic,
            derivative_inverse_analytic=derivative_inverse_analytic,
            eps=eps,
            difference_forward=difference_forward,
            difference_inverse=difference_inverse,
        )

    def bulk_modulus_co2_brine_mixture(self, saturation_co2, equation=None, e=None):
        if equation is None:
            equation = self._version_fluid_mixtures
        if e is None:
            e = self._brie_exponent
        if equation == "wood":
            return 1 / (saturation_co2 / self._k_co2 + (1 - saturation_co2) / self._k_brine)
        if equation == "domenico":
            return saturation_co2 * self._k_co2 + (1 - saturation_co2) * self._k_brine
        if equation == "brie":
            return self._k_co2 + (self._k_brine - self._k_co2) * (1 - saturation_co2) ** e
        raise ValueError("Equation must be either wood, domenico or brie")

    def density_co2_brine_mixture(self, saturation_co2):
        return self._rho_co2 * saturation_co2 + self._rho_brine * (1 - saturation_co2)

    def density_rock(self, rho_fluid, phi=None):
        if phi is None:
            phi = self._phi
        return (1 - phi) * self._rho_matrix + phi * rho_fluid

    def gassmanns_equation_ivanova(self, K_fluid, phi=None):
        if phi is None:
            phi = self._phi
        return self._k_dry + (1 - self._k_dry / self._k_matrix) ** 2 / (
            phi / K_fluid + (1 - phi) / self._k_matrix - self._k_dry / (self._k_matrix**2)
        )

    def gassmanns_equation_hs(self, K_fluid, phi=None):
        if phi is None:
            phi = self._phi
        l_value = (self._k_dry / (self._k_matrix - self._k_dry)) + (K_fluid / (phi * (self._k_matrix - K_fluid)))
        return (l_value / (1 + l_value)) * self._k_matrix

    def gassmanns_equation(self, K_fluid, phi=None, version=None):
        if phi is None:
            phi = self._phi
        if version is None:
            version = self._version_gassmann
        if version == "ivanova":
            return self.gassmanns_equation_ivanova(K_fluid, phi)
        if version == "hs":
            return self.gassmanns_equation_hs(K_fluid, phi)
        raise ValueError("Version must be either ivanova or hs")

    def vs_to_shearmodulus(self, correction_factor=1e-3):
        _ = self._s_wave_velocity**2 * self._rho_ivanova * correction_factor
        return 8.1

    def bulk_modulus_to_vp(self, K, rho):
        my = self.vs_to_shearmodulus()
        return np.sqrt((K + 4 / 3 * my) / rho)

    def saturation_to_vp(self, saturation, correction_factor=None):
        if correction_factor is None:
            correction_factor = self._correction_factor
        k_fluid = self.bulk_modulus_co2_brine_mixture(saturation_co2=saturation)
        k_replaced = self.gassmanns_equation(K_fluid=k_fluid)
        rho_fluid_replaced = self.density_co2_brine_mixture(saturation_co2=saturation)
        rho_rock_replaced = self.density_rock(rho_fluid=rho_fluid_replaced)
        return self.bulk_modulus_to_vp(K=k_replaced, rho=rho_rock_replaced) * correction_factor

    def vp_to_saturation(self, vp, maximum_saturation=None, n_saturation=None, correction_factor=None):
        if maximum_saturation is None:
            maximum_saturation = self._maximum_saturation
        if n_saturation is None:
            n_saturation = self._n_saturation
        if correction_factor is None:
            correction_factor = self._correction_factor
        saturation_vector = np.linspace(0, maximum_saturation, int(n_saturation))
        vp_vector = self.saturation_to_vp(saturation_vector, correction_factor=correction_factor)
        return np.interp(vp, np.flip(vp_vector), np.flip(saturation_vector))


class ArchieTransformation(Transformation):
    """Archie-based saturation and resistivity transformation.

    The class encapsulates the Archie law and its inverse for a fixed fluid
    resistivity, porosity, cementation exponent, and saturation exponent.
    Instances can be used directly as paired forward/backward transformations.

    Parameters
    ----------
    r_fluid : float
        Fluid resistivity used in Archie`s law.
    phi : float
        Porosity of the rock.
    m : float
        Cementation exponent.
    n : float
        Saturation exponent.
    eps : float
        Finite-difference step size for numerical derivatives.
    derivative_function_analytic : callable | None
        Analytic derivative of the forward transformation. If provided, the
        forward derivative uses this callable instead of finite differences.
    derivative_inverse_analytic : callable | None
        Analytic derivative of the inverse transformation. If provided, the
        inverse derivative uses this callable instead of finite differences.
    difference_forward : str
        Finite-difference scheme for the forward derivative. Possible values
        are ``"forward"`` and ``"backward"``. Ignored when an analytic
        derivative is supplied.
    difference_inverse : str
        Finite-difference scheme for the inverse derivative. Possible values
        are ``"forward"`` and ``"backward"``. Ignored when an analytic
        derivative is supplied.
    """

    def __init__(
        self,
        r_fluid=R_FLUID,
        phi=PHI,
        m=CEMENTATION_EXPONENT,
        n=SATURATION_EXPONENT,
        eps=1e-4,
        derivative_function_analytic=None,
        derivative_inverse_analytic=None,
        difference_forward="forward",
        difference_inverse="forward",
    ):
        """Create an Archie transformation instance.

        Parameters
        ----------
        r_fluid : float
            Fluid resistivity used in Archie`s law.
        phi : float
            Porosity of the rock.
        m : float
            Cementation exponent.
        n : float
            Saturation exponent.
        eps : float
            Finite-difference step size for numerical derivatives.
        derivative_function_analytic : callable | None
            Analytic derivative of the forward transformation. If provided,
            it replaces the finite-difference approximation.
        derivative_inverse_analytic : callable | None
            Analytic derivative of the inverse transformation. If provided,
            it replaces the finite-difference approximation.
        difference_forward : str
            Finite-difference scheme for the forward derivative. Possible
            values are ``"forward"`` and ``"backward"``.
        difference_inverse : str
            Finite-difference scheme for the inverse derivative. Possible
            values are ``"forward"`` and ``"backward"``.
        """
        self._r_fluid = r_fluid
        self._phi = phi
        self._m = m
        self._n = n

        super().__init__(
            forward=self.sat_to_res_function,
            inverse=self.res_to_sat_function,
            derivative_function_analytic=derivative_function_analytic,
            derivative_inverse_analytic=derivative_inverse_analytic,
            eps=eps,
            difference_forward=difference_forward,
            difference_inverse=difference_inverse,
        )

    def sat_to_res_function(self, saturation_co2):
        saturation_water = 1 - saturation_co2
        return self._r_fluid * self._phi**-self._m * saturation_water**-self._n

    def res_to_sat_function(self, resistivity):
        saturation_water = (resistivity / (self._r_fluid * self._phi**-self._m)) ** (-1 / self._n)
        return 1 - saturation_water