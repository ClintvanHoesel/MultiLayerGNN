"""Physical unit conversion and physical constants for ``morphology_gnn``.

This module provides a strict singleton utility class :class:`Units` for
performing physical unit conversions and retrieving physical constants. It is
aimed at scientific computing and, in particular, at molecular modeling and
quantum chemistry (Hartree, Bohr, electron charges, ...).

Supported quantities and units
------------------------------
* ``constants``: ``speed_of_light`` (``c``), ``electron_charge`` (``e``),
  ``Avogadro_constant`` (``NA``), ``Bohr_radius``, ``Boltzmann`` (``k_B``),
  ``vacuum_electric_permittivity``.
* ``distance``: ``Angstrom``/``A``/``Ang``, ``Bohr``/``au``/``a.u.``, ``nm``,
  ``pm``, ``m``.
* ``reciprocal distance``: ``1/Angstrom``/``1/A``/``A^-1``/``Angstrom^-1``,
  ``1/Bohr``/``Bohr^-1``, ``1/m``/``m^-1``.
* ``angle``: ``degree``/``deg``, ``radian``/``rad``, ``grad``, ``circle``.
* ``charge``: ``coulomb``/``C``, ``e``.
* ``energy``: ``au``/``a.u.``/``Hartree``, ``eV``, ``kcal/mol``, ``kJ/mol``,
  ``cm^-1``/``cm-1``, ``K``/``Kelvin``, ``Hz``/``Hertz``, ``THz``, ``J``.
* ``dipole``: ``au``/``e*bohr``, ``Debye``/``D``, and every charge unit times
  every distance unit (e.g. ``eA``, ``e*A``, ``Cm``, ``C*m``).
* ``molecular_polarizability``: ``au``/``(e*bohr)^2/hartree``,
  ``e*A^2/V``/``(e*A)^2/eV``, ``C*m^2/V``, ``cm^3``, ``bohr^3``, ``A^3``.
* ``forces``: every energy unit divided by ``Angstrom`` or ``Bohr`` (e.g.
  ``eV/Angstrom``, ``hartree/bohr``).
* ``hessian``: every energy unit divided by ``Angstrom^2`` or ``Bohr^2``.
* ``stress``/pressure: every energy unit divided by ``Angstrom^3``/``Bohr^3``,
  plus ``Pa``, ``GPa``, ``bar`` and ``atm``.

The class is a singleton by convention: do not instantiate it, call the class
methods instead. The constructor raises :class:`ValueError`.

Data provenance
---------------
The constants and conversion factors below are the 2022 CODATA recommended
values published by NIST (https://physics.nist.gov/cuu/Constants/), the newest
available adjustment (released May 2024). ``speed_of_light``, ``electron_charge``,
``Avogadro_constant`` and ``Boltzmann`` have been exact by definition since the
2019 SI redefinition. All values are physical facts, reproduced as-is.
"""

import collections.abc
import math
from typing import Dict, Optional, TypeVar

import numpy

__all__ = ["Units"]


T = TypeVar("T")


class Units:
    """A singleton container for physical constants and unit conversion factors.

    Every supported quantity is a class-level dictionary mapping a unit string
    (e.g. ``"eV"``, ``"Angstrom"``) to its conversion factor relative to that
    quantity's base unit (``Angstrom`` for distance, ``Hartree`` for energy,
    ...). Derived quantities (dipole moments, forces, hessian, stress,
    molecular polarizability) are precomputed at class-definition time from the
    base ``energy`` / ``distance`` / ``charge`` / ... dictionaries. A lowercased
    master index, :attr:`quantities_for_unit`, maps every unit string to the
    set of quantities it belongs to, so conversions can be looked up quickly.

    All functionality is accessed through the class methods; the constructor
    is disabled on purpose. Example::

        >>> Units.convert(123, "angstrom", "bohr")
        232.436313487
        >>> Units.conversion_ratio("kcal/mol", "kJ/mol")
        4.184
    """

    # ------------------------------------------------------------------ #
    # Physical constants (2022 CODATA). Values are physical facts.
    # ------------------------------------------------------------------ #
    constants: Dict[str, float] = {}
    constants["Bohr_radius"] = 0.529177210544  # A
    constants["Avogadro_constant"] = constants["NA"] = 6.02214076e23  # 1/mol
    constants["speed_of_light"] = constants["c"] = 299792458  # m/s
    constants["electron_charge"] = constants["e"] = 1.602176634e-19  # C
    constants["Boltzmann"] = constants["k_B"] = 1.380649e-23  # J/K
    constants["vacuum_electric_permittivity"] = 8.8541878128e-12  # F/m = C/(V*m)

    # ------------------------------------------------------------------ #
    # Base quantities. Each value is the factor relative to the quantity's
    # base unit (e.g. Angstrom for distance, Hartree for energy).
    # ------------------------------------------------------------------ #
    distance: Dict[str, float] = {}
    distance["A"] = distance["Angstrom"] = distance["Ang"] = 1.0
    distance["Bohr"] = distance["bohr"] = distance["a.u."] = distance["au"] = (
        1.0 / constants["Bohr_radius"]
    )
    distance["nm"] = distance["A"] / 10.0
    distance["pm"] = distance["A"] * 100.0
    distance["m"] = distance["A"] * 1e-10

    rec_distance: Dict[str, float] = {}
    rec_distance["1/A"] = rec_distance["1/Ang"] = rec_distance["1/Angstrom"] = (
        rec_distance["A^-1"]
    ) = rec_distance["Ang^-1"] = rec_distance["Angstrom^-1"] = 1.0
    rec_distance["1/m"] = rec_distance["m^-1"] = 1e10
    rec_distance["1/Bohr"] = rec_distance["Bohr^-1"] = constants["Bohr_radius"]

    energy: Dict[str, float] = {}
    energy["au"] = energy["a.u."] = energy["Hartree"] = energy["Ha"] = 1.0
    energy["eV"] = 27.211386245981
    energy["kJ/mol"] = 4.3597447222060e-21 * constants["NA"]
    energy["J"] = 4.3597447222060e-18
    energy["kcal/mol"] = energy["kJ/mol"] / 4.184
    energy["cm^-1"] = energy["cm-1"] = 219474.63136320
    energy["K"] = energy["Kelvin"] = energy["J"] / constants["k_B"]
    energy["Hz"] = energy["Hertz"] = 6.5796839204999e15
    energy["THz"] = energy["Hz"] / 1e12

    mass: Dict[str, float] = {}
    mass["au"] = mass["a.u."] = mass["amu"] = 1.0
    mass["kg"] = 1.66053906892e-27
    mass["g"] = mass["kg"] * 1e3

    time: Dict[str, float] = {}
    time["s"] = 1.0
    time["ms"] = time["s"] * 1e3
    time["us"] = time["s"] * 1e6
    time["ns"] = time["s"] * 1e9
    time["ps"] = time["s"] * 1e12
    time["fs"] = time["s"] * 1e15
    time["au"] = time["a.u."] = time["s"] / 2.4188843265864e-17

    angle: Dict[str, float] = {}
    angle["degree"] = angle["deg"] = 1.0
    angle["radian"] = angle["rad"] = math.pi / 180.0
    angle["grad"] = 100.0 / 90.0
    angle["circle"] = 1.0 / 360.0

    charge: Dict[str, float] = {}
    charge["a.u."] = charge["au"] = charge["e"] = 1.0
    charge["C"] = charge["coulomb"] = constants["e"]

    # ------------------------------------------------------------------ #
    # Derived quantities.
    # ------------------------------------------------------------------ #
    dipole: Dict[str, float] = {}
    for k, v in charge.items():
        if k in ("au", "a.u."):  # reserved for the atomic-unit alias below
            continue
        for k1, v1 in distance.items():
            if k1 in ("au", "a.u."):
                continue
            dipole[k + "*" + k1] = v * v1
            dipole[k + k1] = v * v1
    dipole["au"] = dipole["a.u."] = dipole["e*bohr"]
    dipole["debye"] = dipole["D"] = dipole["Cm"] * constants["c"] * 1e21

    molecular_polarizability: Dict[str, float] = {}
    molecular_polarizability["au"] = molecular_polarizability["a.u."] = (
        molecular_polarizability["e^2*bohr^2/hartree"]
    ) = molecular_polarizability["(e*bohr)^2/hartree"] = 1.0
    molecular_polarizability["e*A^2/V"] = molecular_polarizability["e^2*A^2/eV"] = (
        molecular_polarizability["(e*A)^2/eV"]
    ) = (constants["Bohr_radius"] ** 2 / energy["eV"])
    molecular_polarizability["C*m^2/V"] = (
        molecular_polarizability["e*A^2/V"] * 1e-20 * constants["e"]
    )
    molecular_polarizability["cm^3"] = (
        molecular_polarizability["C*m^2/V"]
        / (4 * numpy.pi * constants["vacuum_electric_permittivity"])
        * 1e6
    )
    molecular_polarizability["A^3"] = molecular_polarizability["Ang^3"] = (
        molecular_polarizability["Angstrom^3"]
    ) = (molecular_polarizability["cm^3"] * 1e24)
    molecular_polarizability["bohr^3"] = (
        molecular_polarizability["Ang^3"] / constants["Bohr_radius"] ** 3
    )

    forces: Dict[str, float] = {}
    hessian: Dict[str, float] = {}
    stress: Dict[str, float] = {}
    for k, v in energy.items():
        for k1, v1 in distance.items():
            forces[k + "/" + k1] = v / v1
            hessian[k + "/" + k1 + "^2"] = v / v1**2
            stress[k + "/" + k1 + "^3"] = v / v1**3
    forces["au"] = forces["a.u."] = forces["Ha/bohr"]
    hessian["au"] = hessian["a.u."] = hessian["Ha/bohr^2"]
    stress["au"] = stress["a.u."] = stress["Ha/bohr^3"]
    stress["Pa"] = stress["J/m^3"]
    stress["GPa"] = stress["Pa"] * 1e-9
    stress["bar"] = stress["Pa"] * 1e-5
    stress["atm"] = stress["bar"] / 1.01325

    # ------------------------------------------------------------------ #
    # Registry of all quantity dictionaries, plus the lowercased master
    # index used for fast lookups in conversion_ratio().
    # ------------------------------------------------------------------ #
    dicts: Dict[str, Dict[str, float]] = {
        "distance": distance,
        "energy": energy,
        "mass": mass,
        "time": time,
        "angle": angle,
        "dipole": dipole,
        "reciprocal distance": rec_distance,
        "forces": forces,
        "hessian": hessian,
        "stress": stress,
        "charge": charge,
        "molecular_polarizability": molecular_polarizability,
    }

    # Lowercased unit string -> {quantity name: conversion factor}.
    quantities_for_unit: Dict[str, Dict[str, float]] = {}
    for quantity in dicts:
        for unit, factor in dicts[quantity].items():
            unit = unit.lower()
            if unit not in quantities_for_unit:
                quantities_for_unit[unit] = {}
            quantities_for_unit[unit][quantity] = factor

    def __init__(self) -> None:
        raise ValueError("Instances of Units cannot be created")

    @classmethod
    def find_unit(cls, unit: str) -> Dict[str, str]:
        """Return ``{quantity name: canonical unit string}`` for *unit*.

        The lookup is case-insensitive. An empty dict is returned when *unit*
        is not a known unit at all.
        """
        ret: Dict[str, str] = {}
        u = unit.lower()
        quantities = cls.quantities_for_unit.get(u, {})
        for quantity in quantities:
            for k in cls.dicts[quantity]:
                if k.lower() == u:
                    ret[quantity] = k
                    break
        return ret

    @classmethod
    def conversion_ratio(cls, inp: str, out: str) -> float:
        """Return the conversion ratio from unit *inp* to *out*.

        Multiplying a value expressed in *inp* by the returned ratio yields the
        same value expressed in *out*. Raises :class:`ValueError` for
        unsupported units or a mismatch of physical dimensions.
        """
        if inp == out:
            return 1.0
        inps = cls.quantities_for_unit.get(inp.lower(), {})
        outs = cls.quantities_for_unit.get(out.lower(), {})
        common = set(inps.keys()) & set(outs.keys())
        if len(common) > 0:
            quantity = common.pop()
            return outs[quantity] / inps[quantity]
        if len(inps) == 0 and len(outs) == 0:
            raise ValueError(f"Unsupported units: '{inp}' and '{out}'")
        if len(inps) > 0 and len(outs) > 0:
            raise ValueError(
                f"Invalid unit conversion: '{inp}' is a unit of "
                f"{', '.join(list(inps.keys()))} and '{out}' is a unit of "
                f"{', '.join(list(outs.keys()))}"
            )
        # Exactly one of the two units is unsupported; give a precise hint.
        invalid, nonempty = (out, inps) if len(inps) else (inp, outs)
        if len(nonempty) == 1:
            quantity = list(nonempty.keys())[0]
            raise ValueError(
                f"Invalid unit conversion: {invalid} is not supported. "
                f"Supported units for {quantity}: "
                f"{', '.join(list(cls.dicts[quantity].keys()))}"
            )
        raise ValueError(
            f"Invalid unit conversion: {invalid} is not a supported unit for "
            f"{', '.join(list(nonempty.keys()))}"
        )

    @classmethod
    def convert(cls, value: T, inp: str, out: str) -> T:
        """Convert *value* from unit *inp* to *out*.

        *value* may be a single number or a (possibly nested) container such
        as a list, tuple or ``numpy`` array. Containers are converted
        recursively, and the result keeps the container type (lists stay
        lists, ``numpy`` arrays stay ``numpy`` arrays, ...). Strings, booleans
        and ``None`` are returned unchanged, as is any value when ``inp == out``.
        """
        if value is None or isinstance(value, (bool, str)) or inp == out:
            return value
        if isinstance(value, collections.abc.Iterable):
            t = type(value)
            if t == numpy.ndarray:
                t = numpy.array  # type: ignore[assignment]
            v = [cls.convert(i, inp, out) for i in value]
            return t(v)  # type: ignore[call-arg,return-value]
        if isinstance(value, (int, float, numpy.generic)):
            return value * cls.conversion_ratio(inp, out)  # type: ignore[operator,return-value]
        return value

    @classmethod
    def ascii2unicode(cls, string: Optional[str]) -> str:
        """Replace ASCII unit notation with Unicode symbols (``^2`` -> ``²``).

        Returns an empty string when ``None`` is passed in.
        """
        if string is None:
            return ""
        return (
            string.replace("^-1", "⁻¹")
            .replace("angstrom", "Å")
            .replace("^2", "²")
            .replace("^3", "³")
            .replace("degree", "°")
            .replace("deg.", "°")
            .replace("Ang", "Å")
            .replace("*", "⋅")
        )
