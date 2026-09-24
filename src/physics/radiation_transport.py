"""
Модуль строгого численного моделирования радиационного переноса,
дифференциальных сечений рассеяния и торможения бета-излучения.
Использует специализированные численные решатели SciPy (scipy.integrate.quad, solve_ivp).
"""

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from typing import Callable, Dict, Tuple, Optional


# Фундаментальные физические константы (CODATA 2022 / NIST)
M_E_C2_KEV = 510.998950        # Энергия покоя электрона (кэВ)
M_U_C2_KEV = 931494.0038       # Энергия покоя атомной единицы массы (кэВ/а.е.м.)
ALPHA_QED = 1.0 / 137.035999   # Постоянная тонкой структуры
PI_R0_SQ_BARNS = 0.2494        # Префактор сечения pi * r_e^2 в барнах (1 барн = 10^-24 см^2)

# Параметры бета-активных радионуклидов
ISOTOPE_DECAY_PARAMS = {
    "Ni-63": {"z_daughter": 29, "e_max_kev": 66.9, "e_avg_tabulated_kev": 17.4, "t_half_years": 100.1},
    "H-3":   {"z_daughter": 2,  "e_max_kev": 18.6, "e_avg_tabulated_kev": 5.7,  "t_half_years": 12.32},
    "C-14":  {"z_daughter": 7,  "e_max_kev": 156.5, "e_avg_tabulated_kev": 49.5, "t_half_years": 5730.0},
    "Pm-147":{"z_daughter": 62, "e_max_kev": 224.0, "e_avg_tabulated_kev": 62.0, "t_half_years": 2.62},
}


def relativistic_fermi_beta_spectrum(energy_kev: float, isotope: str = "Ni-63") -> float:
    """
    Релятивистская дифференциальная вероятность испускания бета-электрона P(E) dE
    с кулоновским фактором Ферми F(Z_d, W) для разрешенных переходов.
    """
    if isotope not in ISOTOPE_DECAY_PARAMS:
        raise ValueError(f"Неизвестный изотоп: {isotope}")
    
    params = ISOTOPE_DECAY_PARAMS[isotope]
    e_max = params["e_max_kev"]
    z_d = params["z_daughter"]
    
    if energy_kev <= 0.0 or energy_kev >= e_max:
        return 0.0
    
    # Полная релятивистская энергия W и максимальная энергия W_0 в единицах m_e*c^2
    w = 1.0 + energy_kev / M_E_C2_KEV
    w0 = 1.0 + e_max / M_E_C2_KEV
    p = np.sqrt(np.maximum(w**2 - 1.0, 1e-12))
    
    # Релятивистский кулоновский параметр Зоммерфельда
    eta = ALPHA_QED * z_d * w / p
    
    # Релятивистская функция Ферми F(Z_d, W)
    if eta > 0.0:
        fermi_factor = 2.0 * np.pi * eta / (1.0 - np.exp(-2.0 * np.pi * eta))
    else:
        fermi_factor = 1.0
        
    # Дифференциальная вероятность dP/dE ~ F(Z,W) * p * W * (W0 - W)^2
    return float(fermi_factor * p * w * (w0 - w)**2)


def compute_normalized_fermi_spectrum(isotope: str = "Ni-63") -> Tuple[Callable[[float], float], float, float]:
    """
    Численно интегрирует спектр Ферми методом scipy.integrate.quad,
    возвращая нормированную функцию плотности вероятности P_norm(E),
    константу нормировки и среднюю энергию бета-спектра <E_beta>.
    """
    if isotope not in ISOTOPE_DECAY_PARAMS:
        raise ValueError(f"Неизвестный изотоп: {isotope}")
    
    e_max = ISOTOPE_DECAY_PARAMS[isotope]["e_max_kev"]
    
    # Численный интеграл нормировки
    norm_const, _ = integrate.quad(
        lambda e: relativistic_fermi_beta_spectrum(e, isotope),
        0.001,
        e_max,
        limit=250,
        epsabs=1e-9,
        epsrel=1e-7
    )
    
    if norm_const <= 0.0:
        norm_const = 1.0
        
    def pdf(e: float) -> float:
        return relativistic_fermi_beta_spectrum(e, isotope) / norm_const
    
    # Численный интеграл средней кинетической энергии <E> = int(E * P(E) dE)
    mean_energy_kev, _ = integrate.quad(
        lambda e: e * pdf(e),
        0.001,
        e_max,
        limit=250,
        epsabs=1e-9,
        epsrel=1e-7
    )
    
    return pdf, norm_const, mean_energy_kev


def joy_luo_stopping_power_exact(
    energy_kev: float, 
    density_g_cm3: float, 
    z_eff: float, 
    a_eff: float
) -> float:
    """
    Удельная тормозная способность dE/dx (кэВ/мкм) по канонической формуле Джоя-Ло (1989)
    с проверенным физическим коэффициентом 7.85 кэВ/мкм:
    -dE/dx = 7.85 * (rho * Z / A) / E * ln(1.166 * (E + 0.73 J) / J)
    где J = 11.5 * 10^-3 * Z (кэВ).
    """
    if energy_kev <= 0.01 or density_g_cm3 <= 0.0 or a_eff <= 0.0:
        return 0.0
    
    e = max(energy_kev, 0.02)
    j = 11.5 * z_eff * 1e-3  # кэВ
    k = 0.73
    argument = max(1.166 * (e + k * j) / j, 1.001)
    
    prefactor = 7.85 * (density_g_cm3 * z_eff / a_eff) / e
    return float(prefactor * np.log(argument))


def solve_csda_range_integral(
    energy_kev: float,
    density_g_cm3: float,
    z_eff: float,
    a_eff: float,
    cutoff_energy_kev: float = 0.05
) -> float:
    """
    Численный интеграл полного пробега в приближении непрерывного замедления (CSDA):
    R_CSDA(E0) = int_{E_cut}^{E0} 1 / S(E) dE   (в мкм)
    Решается адаптивным квадратурным методом scipy.integrate.quad.
    """
    if energy_kev <= cutoff_energy_kev or density_g_cm3 <= 0.0:
        return 0.0
    
    def integrand(e: float) -> float:
        sp = joy_luo_stopping_power_exact(e, density_g_cm3, z_eff, a_eff)
        return 1.0 / sp if sp > 1e-6 else 0.0
    
    res, _ = integrate.quad(
        integrand,
        cutoff_energy_kev,
        energy_kev,
        limit=200,
        epsabs=1e-7,
        epsrel=1e-5
    )
    return float(res)


def solve_energy_loss_ode(
    initial_energy_kev: float,
    density_g_cm3: float,
    z_eff: float,
    a_eff: float,
    max_distance_um: float = 50.0,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Решает дифференциальное уравнение торможения электрона:
    dE(x)/dx = -S(E(x)),  E(0) = E0
    с помощью scipy.integrate.solve_ivp (метод Рунге-Кутты RK45) с терминацией при E(x) -> E_cut.
    """
    cutoff_e = 0.05  # кэВ
    
    def ode_system(x: float, y: np.ndarray) -> np.ndarray:
        e_curr = y[0]
        if e_curr <= cutoff_e:
            return np.array([0.0])
        sp = joy_luo_stopping_power_exact(e_curr, density_g_cm3, z_eff, a_eff)
        return np.array([-sp])
    
    def stopping_event(x: float, y: np.ndarray) -> float:
        return y[0] - cutoff_e
    
    stopping_event.terminal = True
    stopping_event.direction = -1
    
    x_span = (0.0, max_distance_um)
    x_eval = np.linspace(0.0, max_distance_um, num_points)
    
    sol = integrate.solve_ivp(
        ode_system,
        x_span,
        [initial_energy_kev],
        events=stopping_event,
        t_eval=x_eval,
        method="RK45",
        max_step=0.1
    )
    
    return sol.t, sol.y[0]


def compute_spectrum_averaged_csda_range(
    isotope: str,
    density_g_cm3: float,
    z_eff: float,
    a_eff: float
) -> float:
    """
    Вычисляет средневзвешенный по спектру Ферми CSDA-пробег бета-электронов:
    <R_CSDA> = int_0^{E_max} R_CSDA(E) * P(E) dE   (в мкм)
    """
    if isotope not in ISOTOPE_DECAY_PARAMS:
        raise ValueError(f"Неизвестный изотоп: {isotope}")
    
    pdf, _, _ = compute_normalized_fermi_spectrum(isotope)
    e_max = ISOTOPE_DECAY_PARAMS[isotope]["e_max_kev"]
    
    def integrand(e: float) -> float:
        p = pdf(e)
        if p <= 0.0:
            return 0.0
        r = solve_csda_range_integral(e, density_g_cm3, z_eff, a_eff)
        return r * p
    
    avg_range_um, _ = integrate.quad(
        integrand,
        0.1,
        e_max,
        limit=200,
        epsabs=1e-6,
        epsrel=1e-4
    )
    return float(avg_range_um)


def mckinley_feshbach_displacement_cross_section(
    energy_kev: float,
    z_target: float,
    a_target: float,
    displacement_threshold_ed_ev: float
) -> float:
    """
    Релятивистское дифференциальное сечение образования дефектов смещения (барны)
    по формуле Мотта в приближении Мак-Кинли и Фешбаха (McKinley-Feshbach, 1948).
    
    Интегрирует d sigma / dT от E_d до T_max(E):
    sigma_d(E) = pi * r_e^2 * Z^2 * ((1 - beta^2) / beta^4) * [
       (T_max / E_d - 1) - beta^2 * ln(T_max / E_d) +
       pi * alpha * Z * beta * (2 * (sqrt(T_max / E_d) - 1) - ln(T_max / E_d))
    ]
    Если T_max(E) <= E_d, возвращает 0.0 барн (радиационная неуязвимость).
    """
    if energy_kev <= 0.0 or displacement_threshold_ed_ev <= 0.0 or a_target <= 0.0:
        return 0.0
    
    # Релятивистские переменные
    w = 1.0 + energy_kev / M_E_C2_KEV
    beta_sq = 1.0 - 1.0 / (w**2)
    if beta_sq <= 1e-9:
        return 0.0
    beta = np.sqrt(beta_sq)
    
    # Максимальная передаваемая кинетическая энергия лобового упругого удара (эВ)
    m_nucleus_kev = a_target * M_U_C2_KEV
    t_max_ev = (2.0 * energy_kev * (energy_kev + 2.0 * M_E_C2_KEV) / m_nucleus_kev) * 1000.0
    
    if t_max_ev <= displacement_threshold_ed_ev:
        return 0.0
    
    x = t_max_ev / displacement_threshold_ed_ev
    term1 = x - 1.0
    term2 = - beta_sq * np.log(x)
    term3 = np.pi * ALPHA_QED * z_target * beta * (2.0 * (np.sqrt(x) - 1.0) - np.log(x))
    
    prefactor = PI_R0_SQ_BARNS * (z_target**2) * (1.0 - beta_sq) / (beta_sq**2)
    sigma_barns = prefactor * (term1 + term2 + term3)
    
    return float(max(sigma_barns, 0.0))


def compute_spectrum_averaged_displacement_cross_section(
    isotope: str,
    z_target: float,
    a_target: float,
    displacement_threshold_ed_ev: float
) -> float:
    """
    Численно интегрирует сечение дефектообразования по всему непрерывному спектру Ферми:
    <sigma_d> = int_{E_thresh}^{E_max} sigma_d(E) * P(E) dE   (в барнах)
    """
    if isotope not in ISOTOPE_DECAY_PARAMS:
        raise ValueError(f"Неизвестный изотоп: {isotope}")
    
    pdf, _, _ = compute_normalized_fermi_spectrum(isotope)
    e_max = ISOTOPE_DECAY_PARAMS[isotope]["e_max_kev"]
    
    def integrand(e: float) -> float:
        p = pdf(e)
        if p <= 0.0:
            return 0.0
        sig = mckinley_feshbach_displacement_cross_section(
            e, z_target, a_target, displacement_threshold_ed_ev
        )
        return sig * p
    
    avg_sigma_barns, _ = integrate.quad(
        integrand,
        0.1,
        e_max,
        limit=200,
        epsabs=1e-9,
        epsrel=1e-5
    )
    return float(avg_sigma_barns)
