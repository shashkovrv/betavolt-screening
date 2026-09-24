import numpy as np
import pandas as pd
import re

# Физические константы 4 главных бета-изотопов мировой бетавольтаики
ISOTOPES = {
    "Ni-63": {
        "name": "Никель-63",
        "avg_energy_kev": 17.4,   # средняя энергия (кэВ)
        "max_energy_kev": 66.9,   # максимальная кинетическая энергия спектра (кэВ)
        "half_life_years": 100.1, # период полураспада (лет)
        "category": "Долгоживущий (промышленный стандарт)"
    },
    "H-3": {
        "name": "Тритий",
        "avg_energy_kev": 5.7,
        "max_energy_kev": 18.6,
        "half_life_years": 12.32,
        "category": "Среднеживущий (коммерческий, City Labs)"
    },
    "C-14": {
        "name": "Углерод-14",
        "avg_energy_kev": 49.5,   # отличная энергия для широкозонников
        "max_energy_kev": 156.5,
        "half_life_years": 5730.0,# практически вечный источник
        "category": "Сверхдолгоживущий (алмазные батареи)"
    },
    "Pm-147": {
        "name": "Прометий-147",
        "avg_energy_kev": 62.0,   # высокая энергия, высокая мощность
        "max_energy_kev": 224.0,
        "half_life_years": 2.62,
        "category": "Высокомощный (исторический, кардиостимуляторы Betacel)"
    }
}

# Атомные массы элементов (а.е.м.) для расчета кинематики столкновений
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'Ar': 39.948,
    'K': 39.098, 'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.95,
    'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82,
    'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Ba': 137.33, 'La': 138.91,
    'Ta': 180.95, 'W': 183.84, 'Pt': 195.08, 'Au': 196.97, 'Pb': 207.2, 'Bi': 208.98
}


def calculate_ehp_energy(band_gap: float) -> float:
    """
    Энергия образования электронно-дырочной пары по правилу Кляйна (в эВ).
    eps_ehp = 2.8 * Eg + 0.5
    """
    if band_gap <= 0:
        return np.nan
    return 2.8 * band_gap + 0.5


def calculate_penetration_depth(density: float, isotope: str = "Ni-63") -> float:
    """
    Оценка средней глубины пробега бета-электрона в материале (в микрометрах, мкм)
    по эмпирической формуле Фельдмана.
    R = 0.04 * (E_kev ^ 1.75) / density
    """
    if density <= 0 or isotope not in ISOTOPES:
        return np.nan
    
    e_kev = ISOTOPES[isotope]["avg_energy_kev"]
    depth_um = (0.04 * (e_kev ** 1.75)) / density
    return depth_um


def calculate_effective_atomic_weight(formula: str) -> float:
    """Определяет средневзвешенную эффективную атомную массу A_eff по формуле."""
    tokens = re.findall(r'([A-Z][a-z]?)([0-9.]*)', str(formula).strip())
    total_atoms = 0.0
    total_mass = 0.0
    for el, count in tokens:
        c = float(count) if count else 1.0
        m = ATOMIC_WEIGHTS.get(el, 30.0)
        total_atoms += c
        total_mass += c * m
    return total_mass / total_atoms if total_atoms > 0 else 30.0


def calculate_max_recoil_energy(e_max_kev: float, a_eff: float) -> float:
    """
    Релятивистская максимальная кинетическая энергия T_max (в эВ),
    передаваемая атому кристаллической решетки с атомной массой A_eff
    при лобовом упругом столкновении с бета-электроном:
    T_max = 2 * E_max * (E_max + 2 * m_e * c^2) / (M_nucleus * c^2)
    где m_e * c^2 = 511.0 кэВ, M_nucleus * c^2 = A_eff * 931494.0 кэВ.
    """
    if a_eff <= 0 or e_max_kev <= 0:
        return 0.0
    m_e_c2 = 510.9989  # кэВ
    m_u_c2 = 931494.0  # кэВ / а.е.м.
    m_target_kev = a_eff * m_u_c2
    t_max_kev = (2.0 * e_max_kev * (e_max_kev + 2.0 * m_e_c2)) / m_target_kev
    return float(t_max_kev * 1000.0)  # перевод в эВ


def calculate_theoretical_efficiency(band_gap: float, fill_factor: float = 0.85) -> float:
    """
    Теоретический предел КПД бетавольтаического преобразования (в процентах %)
    с учетом микротокового диодного насыщения и ограничения собирания носителей (Олсен / Раппапорт).
    
    Для ультраширокозонных диэлектриков (Eg > 5.5 эВ) вводится физическое подавление
    из-за автокомпенсации дефектов, отсутствия биполярного легирования и пренебрежимо малых длин диффузии L_n, L_p.
    """
    if band_gap <= 0:
        return np.nan
    
    eps = calculate_ehp_energy(band_gap)
    # Коэффициент собирания напряжения с физическим спадом для глубоких изоляторов:
    voltage_factor = 0.72 * (1.0 - np.exp(-band_gap / 0.75)) / (1.0 + (band_gap / 5.2) ** 4.5)
    voc = voltage_factor * band_gap
    efficiency = (voc / eps) * fill_factor * 100.0
    return float(np.clip(efficiency, 0.1, 23.5))


def enrich_with_betavoltaic_metrics(df: pd.DataFrame, band_gap_col: str = "band_gap_dft") -> pd.DataFrame:
    """
    Обогащает DataFrame ключевыми бетавольтаическими характеристиками.
    Рассчитывает параметры для ВСЕХ изотопов из словаря ISOTOPES,
    включая релятивистский кинематический порог повреждения T_max.
    """
    print(f"  Расчет специфических бетавольтаических параметров (по колонке {band_gap_col})...")
    df = df.copy()

    # 1. Общие электрофизические свойства полупроводника
    df["eps_ehp_ev"] = df[band_gap_col].apply(calculate_ehp_energy)
    df["theoretical_efficiency_pct"] = df[band_gap_col].apply(calculate_theoretical_efficiency)
    df["Voc_est_v"] = df[band_gap_col] * 0.70 / (1.0 + (df[band_gap_col] / 5.5) ** 4)

    # 2. Эффективная атомная масса
    if "formula" in df.columns:
        a_eff_series = df["formula"].apply(calculate_effective_atomic_weight)
    else:
        a_eff_series = pd.Series(30.0, index=df.index)
    df["a_eff"] = a_eff_series

    # 3. Динамический расчет под КАЖДЫЙ изотоп из словаря
    for iso_key, iso_data in ISOTOPES.items():
        clean_tag = iso_key.replace("-", "")  # например: Ni63, H3, C14, Pm147
        e_ev = iso_data["avg_energy_kev"] * 1000.0
        e_max = iso_data["max_energy_kev"]

        # Число рожденных пар на один электрон данного изотопа
        df[f"carriers_per_electron_{clean_tag}"] = e_ev / df["eps_ehp_ev"]

        # Глубина проникновения электрона (мкм)
        df[f"penetration_depth_um_{clean_tag}"] = df["density"].apply(
            lambda rho, iso=iso_key: calculate_penetration_depth(rho, iso)
        )

        # Максимальная кинетическая энергия отдачи T_max (эВ)
        df[f"t_max_ev_{clean_tag}"] = [
            calculate_max_recoil_energy(e_max, a) for a in a_eff_series
        ]

        # Радиационная иммунность к данному изотопу (T_max < Ed)
        if "ed_est_ev" in df.columns:
            df[f"is_immune_{clean_tag}"] = (df[f"t_max_ev_{clean_tag}"] < df["ed_est_ev"]).astype(int)

        print(f"   Рассчитаны параметры для изотопа: {iso_data['name']} ({iso_key}) [E_max={e_max} кэВ]")

    print("  Все изотопы успешно обсчитаны!")
    return df


if __name__ == "__main__":
    # Тестовый расчет для эталонных полупроводников
    test_mats = [
        ("Si (Кремний)", 1.12, 2.33, 28.085, 13.0),
        ("4H-SiC (Карбид кремния)", 3.25, 3.21, 20.05, 22.0),
        ("GaN (Нитрид галлия)", 3.40, 6.15, 41.86, 20.0),
        ("C (Алмаз)", 5.47, 3.51, 12.011, 40.0),
        ("NaLiB4O7 (Диэлектрик)", 7.50, 2.45, 23.95, 38.0)
    ]
    
    print("=========================================================================================")
    print("  ТЕСТОВЫЙ РАСЧЕТ МОДЕРНИЗИРОВАННОЙ ФИЗИЧЕСКОЙ МОДЕЛИ:")
    print("=========================================================================================")
    print(f"{'Материал':<22} | {'Eg (эВ)':<7} | {'КПД (%)':<8} | {'T_max Ni63 (эВ)':<15} | {'T_max Pm147 (эВ)':<17}")
    print("-" * 80)
    for name, eg, rho, a, ed in test_mats:
        eff = calculate_theoretical_efficiency(eg)
        t_ni = calculate_max_recoil_energy(66.9, a)
        t_pm = calculate_max_recoil_energy(224.0, a)
        print(f"{name:<22} | {eg:<7.2f} | {eff:<8.2f} | {t_ni:<15.2f} | {t_pm:<17.2f}")
    print("=========================================================================================")