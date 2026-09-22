import numpy as np
import pandas as pd

# Физические константы 4 главных бета-изотопов мировой бетавольтаики
ISOTOPES = {
    "Ni-63": {
        "name": "Никель-63",
        "avg_energy_kev": 17.4,   # средняя энергия (кэВ)
        "half_life_years": 100.1, # период полураспада (лет)
        "category": "Долгоживущий (промышленный стандарт)"
    },
    "H-3": {
        "name": "Тритий",
        "avg_energy_kev": 5.7,
        "half_life_years": 12.32,
        "category": "Среднеживущий (коммерческий, City Labs)"
    },
    "C-14": {
        "name": "Углерод-14",
        "avg_energy_kev": 49.5,   # отличная энергия для широкозонников
        "half_life_years": 5730.0,# практически вечный источник
        "category": "Сверхдолгоживущий (алмазные батареи)"
    },
    "Pm-147": {
        "name": "Прометий-147",
        "avg_energy_kev": 62.0,   # высокая энергия, высокая мощность
        "half_life_years": 2.62,
        "category": "Высокомощный (исторический, кардиостимуляторы Betacel)"
    }
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


def calculate_theoretical_efficiency(band_gap: float, fill_factor: float = 0.85) -> float:
    """
    Теоретический предел КПД бетавольтаического преобразования (в процентах %).
    eta = (Voc / eps_ehp) * FF * 100%
    где Voc ~ 0.70 * Eg
    """
    if band_gap <= 0:
        return np.nan
    
    eps = calculate_ehp_energy(band_gap)
    voc = 0.70 * band_gap
    efficiency = (voc / eps) * fill_factor * 100.0
    return efficiency


def enrich_with_betavoltaic_metrics(df: pd.DataFrame, band_gap_col: str = "band_gap_dft") -> pd.DataFrame:
    """
    Обогащает DataFrame ключевыми бетавольтаическими характеристиками.
    Автоматически рассчитывает параметры для ВСЕХ изотопов из словаря ISOTOPES.
    """
    print(f"⚡ Расчет специфических бетавольтаических параметров (по колонке {band_gap_col})...")
    df = df.copy()

    # 1. Общие электрофизические свойства полупроводника
    df["eps_ehp_ev"] = df[band_gap_col].apply(calculate_ehp_energy)
    df["Voc_est_v"] = 0.70 * df[band_gap_col]
    df["theoretical_efficiency_pct"] = df[band_gap_col].apply(calculate_theoretical_efficiency)

    # 2. Динамический расчет под КАЖДЫЙ изотоп из словаря
    for iso_key, iso_data in ISOTOPES.items():
        clean_tag = iso_key.replace("-", "")  # например: Ni63, H3, C14, Pm147
        e_ev = iso_data["avg_energy_kev"] * 1000.0

        # Число рожденных пар на один электрон данного изотопа
        df[f"carriers_per_electron_{clean_tag}"] = e_ev / df["eps_ehp_ev"]

        # Глубина проникновения электрона (мкм)
        df[f"penetration_depth_um_{clean_tag}"] = df["density"].apply(
            lambda rho, iso=iso_key: calculate_penetration_depth(rho, iso)
        )
        print(f"   Рассчитаны параметры для изотопа: {iso_data['name']} ({iso_key})")

    print(" Все изотопы успешно обсчитаны!")
    return df


if __name__ == "__main__":
    # Тестовый расчет для эталонного Кремния (Si): Eg = 1.12 эВ, плотность = 2.33 г/см3
    si_eg = 1.12
    si_density = 2.33
    
    print("=========================================================")
    print(f"  Тестовый расчет для эталонного Кремния (Si):")
    print(f"  Eg = {si_eg} эВ, Плотность = {si_density} г/см³")
    print(f"  Энергия образования пары (Кляйн): {calculate_ehp_energy(si_eg):.2f} эВ")
    print(f"  Теоретический КПД преобразования:  {calculate_theoretical_efficiency(si_eg):.2f} %")
    print("=========================================================")
    print(f"{'Изотоп':<12} | {'Энергия (кэВ)':<14} | {'Пробег (мкм)':<14} | {'Рождено пар / e⁻':<18}")
    print("-" * 65)

    eps_si = calculate_ehp_energy(si_eg)
    for iso_key, iso_data in ISOTOPES.items():
        e_kev = iso_data["avg_energy_kev"]
        depth = calculate_penetration_depth(si_density, iso_key)
        carriers = (e_kev * 1000.0) / eps_si
        print(f"{iso_key:<12} | {e_kev:<14.1f} | {depth:<14.2f} | {int(carriers):<18}")
    print("=========================================================")