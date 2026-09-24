import pytest
import numpy as np
from src.physics.betavoltaics import (
    calculate_ehp_energy,
    calculate_penetration_depth,
    calculate_theoretical_efficiency,
    calculate_effective_atomic_weight,
    calculate_max_recoil_energy,
    ISOTOPES
)


def test_klein_rule_silicon():
    """Проверка правила Кляйна для эталонного кремния (Eg = 1.12 эВ -> ~3.64 эВ)."""
    si_eg = 1.12
    eps = calculate_ehp_energy(si_eg)
    assert pytest.approx(eps, 0.01) == 3.636


def test_penetration_depth_ni63_silicon():
    """Проверка пробега электронов Ni-63 в кремнии (плотность 2.33 -> ~2.54 мкм)."""
    depth = calculate_penetration_depth(density=2.33, isotope="Ni-63")
    assert pytest.approx(depth, 0.1) == 2.54


def test_efficiency_peak_behavior():
    """Проверка формы кривой КПД: пик в области 2.5-3.5 эВ (SiC, GaN), подавление диэлектриков."""
    eff_si = calculate_theoretical_efficiency(1.12)
    eff_sic = calculate_theoretical_efficiency(3.25)
    eff_gan = calculate_theoretical_efficiency(3.40)
    eff_insulator = calculate_theoretical_efficiency(8.00)
    
    assert 12.0 <= eff_si <= 16.0
    assert 16.0 <= eff_sic <= 21.0
    assert 16.0 <= eff_gan <= 21.0
    assert eff_insulator < 6.0
    assert eff_sic > eff_insulator


def test_relativistic_recoil_energy():
    """Проверка релятивистского расчета T_max для Ni-63 и Pm-147."""
    # Ni-63: E_max = 66.9 кэВ
    t_ni_c = calculate_max_recoil_energy(66.9, 12.011)   # Алмаз (C)
    t_ni_si = calculate_max_recoil_energy(66.9, 28.085)  # Кремний (Si)
    
    assert 12.0 <= t_ni_c <= 14.0
    assert 5.0 <= t_ni_si <= 6.5

    # Pm-147: E_max = 224.0 кэВ (более жесткий спектр)
    t_pm_si = calculate_max_recoil_energy(224.0, 28.085)
    assert 20.0 <= t_pm_si <= 23.0


def test_effective_atomic_weight():
    """Проверка расчета средней атомной массы по химической формуле."""
    a_sic = calculate_effective_atomic_weight("SiC")
    assert pytest.approx(a_sic, 0.1) == (28.085 + 12.011) / 2.0
    
    a_gan = calculate_effective_atomic_weight("GaN")
    assert pytest.approx(a_gan, 0.1) == (69.723 + 14.007) / 2.0


def test_all_isotopes_configured():
    """Проверка наличия всех 4 изотопов со всеми необходимыми полями."""
    assert len(ISOTOPES) == 4
    for iso in ["Ni-63", "H-3", "C-14", "Pm-147"]:
        assert iso in ISOTOPES
        assert "avg_energy_kev" in ISOTOPES[iso]
        assert "max_energy_kev" in ISOTOPES[iso]
        assert "half_life_years" in ISOTOPES[iso]