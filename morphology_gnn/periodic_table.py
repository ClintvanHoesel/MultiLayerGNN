"""Elemental data for ``morphology_gnn``.

This module provides a strict singleton utility class :class:`PeriodicTable`
(also exported as ``PT``) that stores and exposes fundamental elemental
properties used across the package -- e.g. the per-element atomic weights
consumed by the mass-weighted center-of-mass helpers in
:mod:`morphology_gnn.data`.

Data provenance
---------------
* **Atomic weights** -- the CIAAW standard atomic weights (report "Atomic
  Weights 2021", presented in the "Standard Atomic Weights" table of the
  CIAAW web site; the 2024 revision of that table is reflected for Gd, Lu,
  Hf and Zr). Elements without a standard atomic weight (radioactive, no
  natural abundance) carry the mass number of their longest-lived isotope.
  Source: https://www.ciaaw.org/atomic-weights.htm
* **Electron affinities** and **first ionization energies** -- the values
  published by PubChem (NLM/NIH) in its periodic table, stored here in
  electron volts (eV).
  Sources: https://pubchem.ncbi.nlm.nih.gov/ptable/electron-affinity/ and
  https://pubchem.ncbi.nlm.nih.gov/ptable/ionization-energy/
* **Radius / connectors / is_metallic / is_electronegative** -- pragmatic
  heuristics used by empirical bond-guessing logic. In particular the
  ``radius`` values are **not** standard covalent or van der Waals radii, and
  the two boolean flags are practical classifications for connectivity
  guessing rather than strict chemical definitions. They are engineering
  defaults, not measured constants, and are not sourced from CIAAW/PubChem.

Each element is one row of :attr:`PeriodicTable.data` with the layout::

    [symbol, mass, radius, connectors, is_metallic, is_electronegative,
     electron_affinity, ionization_potential]

Index ``0`` is a dummy-atom placeholder (``"Xx"``). Masses are stored in
atomic mass units (amu), radii in Angstrom, and electron affinities /
ionization potentials in electron volts (eV). Every unit-bearing getter
accepts a ``unit`` argument and converts the stored value through
:class:`morphology_gnn.units.Units` (``get_mass(..., unit='kg')``,
``get_radius(..., unit='bohr')``, ``get_electron_affinity(..., unit='au')``,
...).

Like :class:`morphology_gnn.units.Units`, this class is a singleton by
convention: it must not be instantiated -- call the class methods instead.
The constructor raises :class:`ValueError`.
"""

from typing import Dict, List, Literal, Optional, TypeVar, Union, overload

import numpy

from .units import Units

__all__ = ["PeriodicTable", "PT"]

T = TypeVar("T")


class PeriodicTable:
    """A singleton container of elemental properties.

    The class-level :attr:`data` list stores, for every element, its symbol,
    atomic weight (amu), empirical bonding radius (Angstrom), connector
    (valency) count, a metallic flag, an electronegative flag, its electron
    affinity (eV) and its first ionization potential (eV). Missing entries
    for electron affinity / ionization potential (many heavy elements) are
    stored as ``None``.

    All functionality is accessed through the class methods; the constructor
    is disabled on purpose. Example::

        >>> PeriodicTable.get_mass("C")
        12.011
        >>> PeriodicTable.get_mass(6, unit="kg")
        1.99449...e-26
        >>> PeriodicTable.get_symbol(20)
        'Ca'
        >>> PeriodicTable.get_electron_affinity("Cl", unit="au")
        0.13292...
    """

    data: List[List] = [
        # [symbol, mass(amu), radius(Å), connectors, metallic, electronegative,
        #  electron_affinity(eV), ionization_potential(eV)]
        ["Xx", 0.0, 0.0, 0, 0, 0, None, None],
        ["H", 1.008, 0.31, 1, 0, 1, 0.754, 13.598],
        ["He", 4.002602, 0.28, 0, 0, 0, None, 24.587],
        ["Li", 6.94, 1.28, 1, 1, 0, 0.618, 5.392],
        ["Be", 9.0121831, 0.96, 2, 1, 0, None, 9.323],
        ["B", 10.81, 0.84, 3, 0, 0, 0.277, 8.298],
        ["C", 12.011, 0.76, 4, 0, 1, 1.263, 11.260],
        ["N", 14.007, 0.71, 3, 0, 1, None, 14.534],
        ["O", 15.999, 0.66, 2, 0, 1, 1.461, 13.618],
        ["F", 18.998403163, 0.57, 1, 0, 1, 3.339, 17.423],
        ["Ne", 20.180, 0.58, 0, 0, 0, None, 21.565],
        ["Na", 22.98976928, 1.66, 1, 1, 0, 0.548, 5.139],
        ["Mg", 24.305, 1.41, 2, 1, 0, None, 7.646],
        ["Al", 26.9815384, 1.21, 3, 1, 0, 0.441, 5.986],
        ["Si", 28.085, 1.11, 4, 1, 0, 1.385, 8.152],
        ["P", 30.973761998, 1.07, 3, 0, 1, 0.746, 10.487],
        ["S", 32.06, 1.05, 2, 0, 1, 2.077, 10.360],
        ["Cl", 35.45, 1.02, 1, 0, 1, 3.617, 12.968],
        ["Ar", 39.95, 1.06, 0, 0, 0, None, 15.760],
        ["K", 39.0983, 2.03, 1, 1, 0, 0.501, 4.341],
        ["Ca", 40.078, 1.76, 2, 1, 0, None, 6.113],
        ["Sc", 44.955907, 1.70, 3, 1, 0, 0.188, 6.561],
        ["Ti", 47.867, 1.60, 4, 1, 0, 0.079, 6.828],
        ["V", 50.9415, 1.53, 5, 1, 0, 0.525, 6.746],
        ["Cr", 51.9961, 1.39, 6, 1, 0, 0.666, 6.767],
        ["Mn", 54.938043, 1.39, 7, 1, 0, None, 7.434],
        ["Fe", 55.845, 1.32, 6, 1, 0, 0.163, 7.902],
        ["Co", 58.933194, 1.26, 5, 1, 0, 0.661, 7.881],
        ["Ni", 58.6934, 1.24, 4, 1, 0, 1.156, 7.640],
        ["Cu", 63.546, 1.32, 3, 1, 0, 1.228, 7.726],
        ["Zn", 65.38, 1.22, 2, 1, 0, None, 9.394],
        ["Ga", 69.723, 1.22, 3, 1, 0, 0.3, 5.999],
        ["Ge", 72.630, 1.20, 4, 1, 0, 1.35, 7.900],
        ["As", 74.921595, 1.19, 3, 0, 0, 0.81, 9.815],
        ["Se", 78.971, 1.20, 2, 0, 1, 2.021, 9.752],
        ["Br", 79.904, 1.20, 1, 0, 1, 3.365, 11.814],
        ["Kr", 83.798, 1.16, 0, 0, 0, None, 14.000],
        ["Rb", 85.4678, 2.20, 1, 1, 0, 0.468, 4.177],
        ["Sr", 87.62, 1.95, 2, 1, 0, None, 5.695],
        ["Y", 88.905838, 1.90, 3, 1, 0, 0.307, 6.217],
        ["Zr", 91.222, 1.75, 4, 1, 0, 0.426, 6.634],
        ["Nb", 92.90637, 1.64, 5, 1, 0, 0.893, 6.759],
        ["Mo", 95.95, 1.54, 6, 1, 0, 0.746, 7.092],
        ["Tc", 98.0, 1.47, 7, 1, 0, 0.55, 7.28],
        ["Ru", 101.07, 1.46, 6, 1, 0, 1.05, 7.361],
        ["Rh", 102.90549, 1.42, 5, 1, 0, 1.137, 7.459],
        ["Pd", 106.42, 1.39, 4, 1, 0, 0.557, 8.337],
        ["Ag", 107.8682, 1.45, 3, 1, 0, 1.302, 7.576],
        ["Cd", 112.414, 1.44, 2, 1, 0, None, 8.994],
        ["In", 114.818, 1.42, 3, 1, 0, 0.3, 5.786],
        ["Sn", 118.710, 1.39, 4, 1, 0, 1.2, 7.344],
        ["Sb", 121.760, 1.39, 3, 1, 0, 1.07, 8.64],
        ["Te", 127.60, 1.38, 2, 0, 1, 1.971, 9.010],
        ["I", 126.90447, 1.39, 1, 0, 1, 3.059, 10.451],
        ["Xe", 131.293, 1.40, 0, 0, 0, None, 12.130],
        ["Cs", 132.90545196, 2.44, 1, 1, 0, 0.472, 3.894],
        ["Ba", 137.327, 2.15, 2, 1, 0, None, 5.212],
        ["La", 138.90547, 2.07, 3, 1, 0, 0.5, 5.577],
        ["Ce", 140.116, 2.04, 3, 1, 0, 0.5, 5.539],
        ["Pr", 140.90766, 2.03, 3, 1, 0, None, 5.464],
        ["Nd", 144.242, 2.01, 3, 1, 0, None, 5.525],
        ["Pm", 145.0, 1.99, 3, 1, 0, None, 5.55],
        ["Sm", 150.36, 1.98, 3, 1, 0, None, 5.644],
        ["Eu", 151.964, 1.98, 3, 1, 0, None, 5.670],
        ["Gd", 157.249, 1.96, 3, 1, 0, None, 6.150],
        ["Tb", 158.925354, 1.94, 3, 1, 0, None, 5.864],
        ["Dy", 162.500, 1.92, 3, 1, 0, None, 5.939],
        ["Ho", 164.930329, 1.92, 3, 1, 0, None, 6.022],
        ["Er", 167.259, 1.89, 3, 1, 0, None, 6.108],
        ["Tm", 168.934219, 1.90, 3, 1, 0, None, 6.184],
        ["Yb", 173.045, 1.87, 3, 1, 0, None, 6.254],
        ["Lu", 174.96669, 1.87, 3, 1, 0, None, 5.426],
        ["Hf", 178.486, 1.75, 4, 1, 0, None, 6.825],
        ["Ta", 180.94788, 1.70, 5, 1, 0, 0.322, 7.89],
        ["W", 183.84, 1.62, 6, 1, 0, 0.815, 7.98],
        ["Re", 186.207, 1.51, 7, 1, 0, 0.15, 7.88],
        ["Os", 190.23, 1.44, 6, 1, 0, 1.1, 8.7],
        ["Ir", 192.217, 1.41, 5, 1, 0, 1.565, 9.1],
        ["Pt", 195.084, 1.36, 4, 1, 0, 2.128, 9.0],
        ["Au", 196.966570, 1.36, 3, 1, 0, 2.309, 9.226],
        ["Hg", 200.592, 1.32, 2, 1, 0, None, 10.438],
        ["Tl", 204.38, 1.45, 3, 1, 0, 0.2, 6.108],
        ["Pb", 207.2, 1.46, 4, 1, 0, 0.36, 7.417],
        ["Bi", 208.98040, 1.48, 3, 1, 0, 0.946, 7.289],
        ["Po", 209.0, 1.40, 2, 1, 0, 1.9, 8.417],
        ["At", 210.0, 1.50, 1, 0, 1, 2.8, 9.5],
        ["Rn", 222.0, 1.50, 0, 0, 0, None, 10.745],
        ["Fr", 223.0, 2.60, 1, 1, 0, 0.47, 3.9],
        ["Ra", 226.0, 2.21, 2, 1, 0, None, 5.279],
        ["Ac", 227.0, 2.15, 3, 1, 0, None, 5.17],
        ["Th", 232.0377, 2.06, 4, 1, 0, None, 6.08],
        ["Pa", 231.03588, 2.00, 5, 1, 0, None, 5.89],
        ["U", 238.02891, 1.96, 6, 1, 0, None, 6.194],
        ["Np", 237.0, 1.90, 7, 1, 0, None, 6.266],
        ["Pu", 244.0, 1.87, 6, 1, 0, None, 6.06],
        ["Am", 243.0, 1.80, 5, 1, 0, None, 5.993],
        ["Cm", 247.0, 1.69, 4, 1, 0, None, 6.02],
        ["Bk", 247.0, 1.68, 3, 1, 0, None, 6.23],
        ["Cf", 251.0, 1.68, 3, 1, 0, None, 6.30],
        ["Es", 252.0, 1.65, 3, 1, 0, None, 6.42],
        ["Fm", 257.0, 1.67, 3, 1, 0, None, 6.50],
        ["Md", 258.0, 1.73, 3, 1, 0, None, 6.58],
        ["No", 259.0, 1.76, 3, 1, 0, None, 6.65],
        ["Lr", 266.0, 1.61, 3, 1, 0, None, None],
        ["Rf", 267.0, 1.57, 4, 1, 0, None, None],
        ["Db", 268.0, 1.49, 5, 1, 0, None, None],
        ["Sg", 269.0, 1.43, 6, 1, 0, None, None],
        ["Bh", 270.0, 1.41, 7, 1, 0, None, None],
        ["Hs", 277.0, 1.34, 6, 1, 0, None, None],
        ["Mt", 278.0, 1.36, 5, 1, 0, None, None],
        ["Ds", 281.0, 1.28, 4, 1, 0, None, None],
        ["Rg", 282.0, 1.30, 3, 1, 0, None, None],
        ["Cn", 285.0, 1.44, 2, 1, 0, None, None],
        ["Nh", 286.0, 1.30, 3, 1, 0, None, None],
        ["Fl", 289.0, 1.28, 4, 1, 0, None, None],
        ["Mc", 290.0, 1.53, 3, 1, 0, None, None],
        ["Lv", 293.0, 1.46, 2, 1, 0, None, None],
        ["Ts", 294.0, 1.45, 1, 0, 1, None, None],
        ["Og", 294.0, 1.40, 0, 0, 0, None, None],
    ]

    # Canonical (capitalized) symbol -> row index (atomic number).
    symtonum: Dict[str, int] = {row[0]: i for i, row in enumerate(data)}

    # Symbols accepted for the different kinds of dummy atoms.
    dummysymbols: List[str] = ["Xx", "El", "Eh", "J"]

    # ------------------------------------------------------------------ #
    # Instantiation is disabled: the class itself is the singleton.
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        raise ValueError("Instances of PeriodicTable cannot be created")

    # ------------------------------------------------------------------ #
    # Symbol <-> atomic number.
    # ------------------------------------------------------------------ #
    @classmethod
    def get_atomic_number(cls, symbol: str) -> int:
        """Convert an element *symbol* (case-insensitive) to its atomic number.

        All dummy-atom symbols (``"Xx"``, ``"El"``, ``"Eh"``, ``"J"``) map to
        ``0``. Raises :class:`ValueError` for unknown symbols.
        """
        key = symbol.lower().capitalize()
        if key in cls.dummysymbols:
            return 0
        try:
            return cls.symtonum[key]
        except KeyError:
            raise ValueError(f"trying to convert incorrect atomic symbol: {symbol!r}")

    @classmethod
    def get_symbol(cls, atnum: int) -> str:
        """Convert an atomic number to its canonical element *symbol*.

        Raises :class:`ValueError` for out-of-range atomic numbers.
        """
        try:
            return cls.data[atnum][0]
        except (IndexError, TypeError):
            raise ValueError(f"trying to convert incorrect atomic number: {atnum!r}")

    # ------------------------------------------------------------------ #
    # Property getters. All accept either a symbol (str) or an atomic number
    # (int / numpy integer), and the unit-bearing ones convert through Units.
    # ------------------------------------------------------------------ #
    @classmethod
    def get_mass(cls, arg: Union[str, int], unit: str = "amu") -> float:
        """Return the atomic weight of *arg* (amu by default).

        The dummy symbols ``"El"`` and ``"Eh"`` are given hydrogen's mass so
        these placeholders carry a physical weight. Any unit supported by
        :class:`morphology_gnn.units.Units` for ``mass`` may be requested,
        e.g. ``unit="kg"``, ``unit="g"`` or ``unit="au"``.
        """
        if isinstance(arg, str) and arg.lower().capitalize() in ("El", "Eh"):
            return cls.get_mass("H", unit=unit)
        return cls._to_unit(cls._get_property(arg, 1), "amu", unit)

    @classmethod
    def get_radius(cls, arg: Union[str, int], unit: str = "angstrom") -> float:
        """Return the empirical bonding radius of *arg* (Angstrom by default).

        Convertible to any :class:`morphology_gnn.units.Units` distance unit
        (``"bohr"``, ``"nm"``, ``"pm"``, ``"m"``, ...).
        """
        return cls._to_unit(cls._get_property(arg, 2), "angstrom", unit)

    @classmethod
    def get_connectors(cls, arg: Union[str, int]) -> int:
        """Return the number of connectors (valency hint) of *arg*."""
        return cls._get_property(arg, 3)

    @classmethod
    def get_metallic(cls, arg: Union[str, int]) -> int:
        """Return the pragmatic metallic flag (0 or 1) of *arg*."""
        return cls._get_property(arg, 4)

    @classmethod
    def get_electronegative(cls, arg: Union[str, int]) -> int:
        """Return the pragmatic electronegative flag (0 or 1) of *arg*."""
        return cls._get_property(arg, 5)

    @classmethod
    def get_electron_affinity(
        cls, arg: Union[str, int], unit: str = "eV"
    ) -> Optional[float]:
        """Return the electron affinity of *arg* in eV (default).

        Returns ``None`` when the value is unavailable. Convertible to any
        :class:`morphology_gnn.units.Units` energy unit (``"au"``, ``"kJ/mol"``,
        ``"kcal/mol"``, ...).
        """
        return cls._to_unit(cls._get_property(arg, 6), "eV", unit)

    @classmethod
    def get_ionization_energy(
        cls, arg: Union[str, int], unit: str = "eV"
    ) -> Optional[float]:
        """Return the first ionization energy of *arg* in eV (default).

        Returns ``None`` when the value is unavailable. Convertible to any
        :class:`morphology_gnn.units.Units` energy unit.
        """
        return cls._to_unit(cls._get_property(arg, 7), "eV", unit)

    # ------------------------------------------------------------------ #
    # Property setters (mutate the class-level data in place).
    # ------------------------------------------------------------------ #
    @classmethod
    def set_mass(cls, element: str, value: float) -> None:
        """Set the atomic weight of *element* (amu) to *value*."""
        cls.data[cls._as_index(element)][1] = value

    @classmethod
    def set_radius(cls, element: str, value: float) -> None:
        """Set the bonding radius of *element* (Angstrom) to *value*."""
        cls.data[cls._as_index(element)][2] = value

    @classmethod
    def set_connectors(cls, element: str, value: int) -> None:
        """Set the connector (valency) count of *element* to *value*."""
        cls.data[cls._as_index(element)][3] = value

    # ------------------------------------------------------------------ #
    # Internal helpers.
    # ------------------------------------------------------------------ #
    @classmethod
    def _as_index(cls, arg: Union[str, int]) -> int:
        """Resolve an element symbol or atomic number to a row index."""
        if isinstance(arg, str):
            return cls.get_atomic_number(arg)
        if isinstance(arg, (int, numpy.integer)):
            if 0 <= arg < len(cls.data):
                return int(arg)
            raise ValueError(f"trying to convert incorrect atomic number: {arg!r}")
        raise ValueError(
            f"expected an element symbol or atomic number, got {type(arg).__name__}"
        )

    @classmethod
    def _to_unit(cls, value: T, unit_in: str, unit_out: str) -> T:
        """Convert *value* from *unit_in* to *unit_out* via :class:`Units`.

        ``None`` values (missing data) pass through unchanged, and values
        already expressed in *unit_out* are returned untouched. The generic
        ``T`` keeps the call-site type: an ``Optional`` column stays
        ``Optional``, an always-``float`` column stays ``float``.
        """
        if value is None or unit_in == unit_out:
            return value
        return value * Units.conversion_ratio(unit_in, unit_out)  # type: ignore

    @classmethod
    @overload
    def _get_property(cls, arg: Union[str, int], prop: Literal[1, 2]) -> float: ...

    @classmethod
    @overload
    def _get_property(cls, arg: Union[str, int], prop: Literal[3, 4, 5]) -> int: ...

    @classmethod
    @overload
    def _get_property(
        cls, arg: Union[str, int], prop: Literal[6, 7]
    ) -> Optional[float]: ...

    @classmethod
    def _get_property(
        cls, arg: Union[str, int], prop: int
    ) -> Union[str, float, int, None]:
        """Return the property stored at column *prop* of the element *arg*.

        Skeleton method behind :meth:`get_mass`, :meth:`get_radius`, ...:
        ``prop`` indexes into the per-element rows of :attr:`data` (``1`` =
        mass, ``2`` = radius, ``3`` = connectors, ``4``/``5`` = flags,
        ``6``/``7`` = electron affinity / ionization potential).
        """
        return cls.data[cls._as_index(arg)][prop]


PT = PeriodicTable
