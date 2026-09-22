import pytest
from src.physics.betavoltaics import calculate_ehp_energy, calculate_penetration_depth, calculate_theoretical_efficiency


def test_klein_rule_silicon():
    """Проверка правила Кляйна для эталонного кремния (Eg = 1.12 эВ -> ~3.64 эВ)."""
    si_eg = 1.12
    eps = calculate_ehp_energy(si_eg)
    assert pytest.approx(eps, 0.05) == 3.636


def test_penetration_depth_ni63_silicon():
    """Проверка пробега электронов Ni-63 в кремнии (плотность 2.33 -> ~2.54 мкм)."""
    depth = calculate_penetration_depth(density=2.33, isotope="Ni-63")
    assert pytest.approx(depth, 0.1) == 2.54


def test_efficiency_bounds():
    """КПД полупроводника с Eg=2.5 эВ должен быть в разумных физических пределах 15-22%."""
    eff = calculate_theoretical_efficiency(band_gap=2.5)
    assert 15.0 <= eff <= 22.0