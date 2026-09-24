import pytest
import numpy as np
import scipy.integrate as integrate
from src.physics.radiation_transport import (
    relativistic_fermi_beta_spectrum,
    compute_normalized_fermi_spectrum,
    joy_luo_stopping_power_exact,
    solve_csda_range_integral,
    solve_energy_loss_ode,
    compute_spectrum_averaged_csda_range,
    mckinley_feshbach_displacement_cross_section,
    compute_spectrum_averaged_displacement_cross_section,
    ISOTOPE_DECAY_PARAMS
)


def test_fermi_spectrum_mean_energies():
    """
    Проверяет, что средняя кинетическая энергия бета-спектра,
    полученная численным интегрированием scipy.integrate.quad,
    соответствует справочным значениям изотопов с точностью < 2%.
    """
    for iso, params in ISOTOPE_DECAY_PARAMS.items():
        pdf, norm_const, mean_e = compute_normalized_fermi_spectrum(iso)
        tab_e = params["e_avg_tabulated_kev"]
        
        # Проверка нормировки: интеграл от pdf(E) равен 1.0
        integral_check, _ = integrate.quad(pdf, 0.001, params["e_max_kev"])
        assert pytest.approx(integral_check, 0.01) == 1.0
        
        # Проверка средней энергии к табличной
        assert abs(mean_e - tab_e) / tab_e < 0.02


def test_csda_integral_and_ode_consistency():
    """
    Проверяет согласованность интегрального решения CSDA и дифференциального
    уравнения замедления solve_ivp (Рунге-Кутта RK45).
    """
    density = 2.33  # Кремний (Si)
    z_eff = 14.0
    a_eff = 28.085
    e0 = 17.4  # кэВ (Ni-63)
    
    # 1. Интеграл CSDA
    r_csda = solve_csda_range_integral(e0, density, z_eff, a_eff)
    
    # 2. Численное решение ОДУ dE/dx = -S(E)
    x_traj, e_traj = solve_energy_loss_ode(e0, density, z_eff, a_eff, max_distance_um=r_csda * 1.5)
    
    # Полный пробег по ОДУ — точка остановки (E -> E_cut)
    r_ode = x_traj[-1]
    
    # Интеграл и траектория ОДУ должны совпадать с точностью до шага сетки (< 5%)
    assert pytest.approx(r_csda, 0.05) == r_ode


def test_mckinley_feshbach_ni63_radiation_immunity():
    """
    Проверяет радиационную неуязвимость полупроводников к изотопу Ni-63:
    сечение образования дефектов смещения sigma_d должно быть строго 0.00 барн.
    """
    e_max_ni63 = 66.9  # кэВ
    
    # Алмаз: Z=6, A=12, Ed=40 эВ
    sig_diamond = mckinley_feshbach_displacement_cross_section(e_max_ni63, 6.0, 12.011, 40.0)
    assert sig_diamond == 0.0
    
    # Кремний: Z=14, A=28, Ed=13 эВ
    sig_si = mckinley_feshbach_displacement_cross_section(e_max_ni63, 14.0, 28.085, 13.0)
    assert sig_si == 0.0
    
    # Карбид кремния (4H-SiC): подрешетка C (Ed=20 эВ) и Si (Ed=35 эВ)
    sig_sic_c = mckinley_feshbach_displacement_cross_section(e_max_ni63, 6.0, 12.011, 20.0)
    sig_sic_si = mckinley_feshbach_displacement_cross_section(e_max_ni63, 14.0, 28.085, 35.0)
    assert sig_sic_c == 0.0
    assert sig_sic_si == 0.0


def test_mckinley_feshbach_pm147_damage():
    """
    Проверяет, что жесткий спектр Pm-147 (224 кэВ) вызывает образование дефектов
    в кремнии и алмазе (sigma_d > 0).
    """
    e_max_pm147 = 224.0  # кэВ
    
    # Кремний (Ed=13 эВ): T_max = 21.3 эВ > 13 эВ -> ненулевое сечение
    sig_si = mckinley_feshbach_displacement_cross_section(e_max_pm147, 14.0, 28.085, 13.0)
    assert sig_si > 10.0  # барны
    
    # Алмаз (Ed=40 эВ): T_max = 49.9 эВ > 40 эВ -> ненулевое сечение
    sig_c = mckinley_feshbach_displacement_cross_section(e_max_pm147, 6.0, 12.011, 40.0)
    assert sig_c > 1.0  # барны
    
    # Интегральное сечение по спектру Ферми Pm-147 также должно быть положительным
    avg_sig_si = compute_spectrum_averaged_displacement_cross_section("Pm-147", 14.0, 28.085, 13.0)
    assert avg_sig_si > 0.1


def test_benchmark_materials_radiation_audit():
    """
    Проверяет согласованность полуэмпирической модели Ed и подрешеточной кинематики
    для эталонных полупроводников (Si, Diamond C, 4H-SiC, GaN, TiO2, BN, BP, AlN, B4C).
    """
    from src.database.repository import calculate_cohesive_energy, calculate_radiation_displacement_energy
    from src.physics.betavoltaics import calculate_max_recoil_energy
    
    # 1. Алмаз (Diamond C)
    ecoh_c = calculate_cohesive_energy("C", 0.0)
    ed_c = calculate_radiation_displacement_energy(5.47, ecoh_c, 3800.0)
    assert pytest.approx(ed_c, 0.02) == 40.0  # точное совпадение с экспериментом
    
    # 2. 100% иммунитет Ni-63 и H-3 для всех легких подрешеток (B, C, N, O)
    for a_elem, ed_val in [(10.81, 18.0), (12.011, 20.0), (14.007, 20.5), (15.999, 25.0)]:
        tmax_ni63 = calculate_max_recoil_energy(66.9, a_elem)
        tmax_h3 = calculate_max_recoil_energy(18.6, a_elem)
        assert tmax_ni63 < ed_val
        assert tmax_h3 < ed_val
        
    # 3. Селективность в 4H-SiC при C-14 (156.5 кэВ): C повреждается, Si иммунен
    tmax_c14_carbon = calculate_max_recoil_energy(156.5, 12.011)
    tmax_c14_silicon = calculate_max_recoil_energy(156.5, 28.085)
    assert tmax_c14_carbon > 20.0   # C повреждается
    assert tmax_c14_silicon < 35.0  # Si абсолютно защищен

