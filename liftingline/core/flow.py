"""Freestream condition: speed, air state and flow angles."""

from __future__ import annotations

from dataclasses import dataclass

from . import atmosphere as atm


@dataclass(frozen=True)
class FlowCondition:
    """Undisturbed flow relative to the wing (SI units).

    ``alpha_deg`` is the angle of attack of the wing's root chord line (before
    root incidence and twist), ``beta_deg`` the sideslip angle (> 0: wind from the
    right). Sideslip is only modelled by the numerical solver.
    """

    V: float  # true airspeed, m/s
    rho: float  # density, kg/m^3
    mu: float  # dynamic viscosity, Pa s
    a: float  # speed of sound, m/s
    alpha_deg: float = 0.0
    beta_deg: float = 0.0
    T: float = atm.T0
    p: float = atm.P0
    altitude_m: float | None = None  # informational (None for manual atmospheres)

    def __post_init__(self) -> None:
        if not self.V > 0.0:
            raise ValueError("airspeed must be positive")
        if not self.rho > 0.0:
            raise ValueError("density must be positive")
        if not self.mu > 0.0:
            raise ValueError("viscosity must be positive")
        if not self.a > 0.0:
            raise ValueError("speed of sound must be positive")

    # ------------------------------------------------------------------ derived
    @property
    def q(self) -> float:
        """Dynamic pressure, Pa."""
        return 0.5 * self.rho * self.V**2

    @property
    def mach(self) -> float:
        return self.V / self.a

    def reynolds(self, length):
        """Reynolds number for a length (scalar or numpy array), m."""
        return self.rho * self.V * length / self.mu

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_isa(
        cls,
        altitude_m: float = 0.0,
        *,
        V: float | None = None,
        mach: float | None = None,
        alpha_deg: float = 0.0,
        beta_deg: float = 0.0,
        delta_T: float = 0.0,
    ) -> "FlowCondition":
        """Flow in the ISA (+``delta_T``) at ``altitude_m``; give either ``V`` or ``mach``."""
        state = atm.isa(altitude_m, delta_T)
        if (V is None) == (mach is None):
            raise ValueError("specify exactly one of V or mach")
        speed = V if V is not None else mach * state.a
        return cls(
            V=speed,
            rho=state.rho,
            mu=state.mu,
            a=state.a,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            T=state.T,
            p=state.p,
            altitude_m=altitude_m,
        )

    @classmethod
    def from_manual(
        cls,
        *,
        rho: float,
        V: float | None = None,
        mach: float | None = None,
        mu: float | None = None,
        T: float | None = None,
        a: float | None = None,
        alpha_deg: float = 0.0,
        beta_deg: float = 0.0,
    ) -> "FlowCondition":
        """Flow from user-entered air properties.

        Missing pieces are derived: temperature from the speed of sound (or the
        standard 288.15 K), viscosity from Sutherland's law at that temperature.
        """
        if a is None:
            temp = T if T is not None else atm.T0
            a_val = atm.speed_of_sound(temp)
        else:
            a_val = a
            temp = T if T is not None else atm.temperature_from_speed_of_sound(a)
        mu_val = mu if mu is not None else atm.sutherland_viscosity(temp)
        if (V is None) == (mach is None):
            raise ValueError("specify exactly one of V or mach")
        speed = V if V is not None else mach * a_val
        return cls(
            V=speed,
            rho=rho,
            mu=mu_val,
            a=a_val,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            T=temp,
            p=rho * atm.R_AIR * temp,
            altitude_m=None,
        )
