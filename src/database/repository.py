import sys
from pathlib import Path

# Гарантируем, что Python видит корень проекта при любом способе запуска
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import re
import sqlite3
import numpy as np
import pandas as pd
from src.database.schema import CREATE_TABLES_SQL
from src.physics.betavoltaics import enrich_with_betavoltaic_metrics


# Фундаментальные энергии когезии простых веществ (эВ/атом, справочник Киттеля / CRC Handbook)
ELEMENTAL_ECOH = {
    'H': 2.27, 'He': 0.00, 'Li': 1.63, 'Be': 3.32, 'B': 5.81, 'C': 7.37, 'N': 4.88, 'O': 2.58, 'F': 0.82, 'Ne': 0.02,
    'Na': 1.11, 'Mg': 1.51, 'Al': 3.39, 'Si': 4.63, 'P': 3.28, 'S': 2.90, 'Cl': 1.25, 'Ar': 0.08,
    'K': 0.93, 'Ca': 1.84, 'Sc': 3.90, 'Ti': 4.85, 'V': 5.31, 'Cr': 4.10, 'Mn': 2.92, 'Fe': 4.28, 'Co': 4.39, 'Ni': 4.44,
    'Cu': 3.49, 'Zn': 1.35, 'Ga': 2.81, 'Ge': 3.85, 'As': 2.96, 'Se': 2.41, 'Br': 1.16, 'Kr': 0.12,
    'Rb': 0.85, 'Sr': 1.72, 'Y': 4.37, 'Zr': 6.25, 'Nb': 7.57, 'Mo': 6.82, 'Tc': 6.85, 'Ru': 6.74, 'Rh': 5.75, 'Pd': 3.89,
    'Ag': 2.95, 'Cd': 1.16, 'In': 2.52, 'Sn': 3.14, 'Sb': 2.75, 'Te': 2.19, 'I': 1.11, 'Xe': 0.16,
    'Cs': 0.80, 'Ba': 1.90, 'La': 4.47, 'Hf': 6.44, 'Ta': 8.10, 'W': 8.90, 'Re': 8.03, 'Os': 8.17, 'Ir': 6.94, 'Pt': 5.84,
    'Au': 3.81, 'Hg': 0.67, 'Tl': 1.88, 'Pb': 2.02, 'Bi': 2.18
}

# Известные температуры плавления/сублимации ключевых полупроводниковых соединений (К)
KNOWN_COMPOUND_TM = {
    'C': 3800.0,
    'BN': 3246.0,
    'AlN': 2470.0,
    'SiC': 3100.0,
    'TiO2': 2116.0,
    'B4C': 2720.0,
    'B6P': 2270.0,
    'GaN': 2773.0,
    'BP': 2270.0,
    'Ga2O3': 2170.0,
    'Si': 1687.0
}


def parse_composition(formula: str) -> dict[str, float]:
    """Парсит химическую формулу и возвращает молярные доли элементов."""
    tokens = re.findall(r'([A-Z][a-z]?)([0-9.]*)', str(formula).strip())
    comp = {}
    for el, count in tokens:
        c = float(count) if count else 1.0
        comp[el] = comp.get(el, 0.0) + c
    total = sum(comp.values()) if comp else 1.0
    return {el: c / total for el, c in comp.items()}


def calculate_cohesive_energy(formula: str, formation_energy_per_atom: float = 0.0) -> float:
    """
    Расчет энергии когезии кристаллической решетки Ecoh (в эВ/атом).
    Ecoh = sum(c_i * Ecoh_elem_i) + |ΔHf|
    """
    comp = parse_composition(formula)
    fe_abs = abs(formation_energy_per_atom) if pd.notna(formation_energy_per_atom) else 0.0
    ecoh_elem = sum(comp.get(el, 3.5) * ELEMENTAL_ECOH.get(el, 3.5) for el in comp)
    return ecoh_elem + fe_abs


def get_effective_melting_temp(formula: str, max_elem_tm: float) -> float:
    """Определяет эффективную температуру термодеструкции/плавления решетки."""
    f_clean = re.sub(r'[^A-Za-z0-9]', '', str(formula).strip())
    if f_clean in KNOWN_COMPOUND_TM:
        return KNOWN_COMPOUND_TM[f_clean]
    if pd.notna(max_elem_tm) and max_elem_tm > 0:
        return float(max_elem_tm)
    return 1500.0


def classify_material_and_viability(formula: str) -> tuple[str, int]:
    """
    Классифицирует кристаллическое соединение по химическому классу
    и определяет его технологическую жизнеспособность для бетавольтаики.
    
    Исключает:
    - Растворимые соли и галогениды (F, Cl, Br, I);
    - Гидриды и гидроксиды (H);
    - Токсичные цианиды (CN) и оксоанионы (CO3, NO3, SO4, PO4);
    - Нестабильные ионные ацетилиды щелочных/щелочноземельных металлов (CaC2, SrC2, LiC);
    - Радиоактивные актиниды и благородные газы.
    """
    f = str(formula).strip()

    # 1. Проверка на нежизнеспособные / нестабильные классы
    if any(h in f for h in ["F", "Cl", "Br", "I"]):
        return "Исключен (Галогенид/Соль)", 0
    if "H" in f:
        return "Исключен (Гидрид/Гидроксид)", 0
    if any(grp in f for grp in ["CN", "CO3", "NO3", "SO4", "PO4", "NH4"]):
        return "Исключен (Сложная соль/Цианид)", 0
    if re.match(r"^(Li|Na|K|Rb|Cs|Ca|Sr|Ba)[0-9]*C[0-9]*$", f):
        return "Исключен (Ионный ацетилид)", 0
    if any(act in f for act in ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "He", "Ne", "Ar", "Kr", "Xe"]):
        return "Исключен (Радиоактивный/Инертный)", 0

    elems = set(re.findall(r"[A-Z][a-z]?", f))

    # 2. Ковалентные полупроводники (IV, III-V, карбиды, бориды, нитриды, фосфиды)
    if ("N" in elems or "C" in elems or "B" in elems or "P" in elems or "Si" in elems or "As" in elems) and \
       ("O" not in elems) and ("S" not in elems) and ("Se" not in elems) and ("Te" not in elems):
        return "Ковалентные (IV, III-V, карбиды, бориды, нитриды)", 1

    # 3. Оксидные полупроводники
    if "O" in elems:
        if len(elems) <= 3:
            return "Оксидные полупроводники (простые и тройные)", 1
        else:
            return "Сложные оксиды", 1

    # 4. Халькогениды (II-VI, слоистые TMDC)
    if any(ch in elems for ch in ["S", "Se", "Te"]):
        if len(elems) <= 3:
            return "Халькогениды (II-VI, дихалькогениды)", 1
        else:
            return "Сложные халькогениды", 1

    return "Прочие полупроводники", 1


def calculate_radiation_displacement_energy(
    band_gap_ev: float,
    cohesive_energy_ev: float,
    melting_temp_k: float
) -> float:
    """
    Полуэмпирическая физическая оценка пороговой энергии смещения атомов Ed (в эВ).
    Основана на расширенной модели Кинчина-Пиза и корреляциях Келли-Гроувса:
    Ed = 8.0 + 1.8 * Eg + 2.0 * Ecoh + 0.002 * Tm
    """
    eg = float(band_gap_ev) if pd.notna(band_gap_ev) else 1.0
    ecoh = float(cohesive_energy_ev) if pd.notna(cohesive_energy_ev) else 4.0
    tm = float(melting_temp_k) if (pd.notna(melting_temp_k) and melting_temp_k > 0) else 1500.0
    
    ed = 8.0 + 1.8 * eg + 2.0 * ecoh + 0.002 * tm
    return float(np.clip(ed, 10.0, 50.0))


def build_database(
    calibrated_path: str = "data/03_features/materials_calibrated.parquet",
    db_path: str = "data/05_database/betavoltaic_library.db"
):
    print(" Формирование итоговой реляционной базы данных...")
    
    if not os.path.exists(calibrated_path):
        raise FileNotFoundError(f"Файл {calibrated_path} не найден! Запусти delta_learner.py.")

    df = pd.read_parquet(calibrated_path)
    print(f"  Загружено материалов: {len(df)}")

    # 1. Химическая классификация и жизнеспособность
    print("  [1/5] Химическая классификация и оценка жизнеспособности...")
    classified = [classify_material_and_viability(f) for f in df["formula"]]
    df["material_class"] = [c[0] for c in classified]
    df["is_viable"] = [c[1] for c in classified]

    # 2. Расчет энергии когезии, энергии смещения Ed и индекса стойкости R_score
    print("  [2/5] Расчет физической энергии когезии Ecoh, пороговой энергии Ed и индекса R_score...")
    tm_col = "MagpieData maximum MeltingT" if "MagpieData maximum MeltingT" in df.columns else "MagpieData mean MeltingT"
    
    df["cohesive_energy_ev"] = [
        calculate_cohesive_energy(f, fe)
        for f, fe in zip(df["formula"], df["formation_energy"])
    ]
    
    df["melting_temp_eff_k"] = [
        get_effective_melting_temp(f, tm)
        for f, tm in zip(df["formula"], df[tm_col] if tm_col in df.columns else [1500.0]*len(df))
    ]

    df["ed_est_ev"] = [
        calculate_radiation_displacement_energy(eg, ecoh, tm)
        for eg, ecoh, tm in zip(df["band_gap_calibrated"], df["cohesive_energy_ev"], df["melting_temp_eff_k"])
    ]

    # Нормировка относительно эталонного алмаза (Ed_diamond = 8.0 + 1.8*5.47 + 2.0*7.37 + 0.002*3800 = 40.186 эВ)
    ed_diamond = 8.0 + 1.8 * 5.47 + 2.0 * 7.37 + 0.002 * 3800.0
    df["radiation_resistance_score"] = (df["ed_est_ev"] / ed_diamond) * 100.0

    # 3. Физический расчет бетавольтаики для 4 изотопов на КАЛИБРОВАННОЙ зоне
    print("  [3/5] Физическое моделирование бетавольтаики для 4 изотопов (Ni-63, H-3, C-14, Pm-147)...")
    df = enrich_with_betavoltaic_metrics(df, band_gap_col="band_gap_calibrated")

    # 4. Инициализация SQLite
    print("  [4/5] Создание реляционных таблиц SQLite...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(CREATE_TABLES_SQL)

    # 5. Запись таблиц
    print("  [5/5] Запись данных в БД...")
    materials_table = df[[
        "mp_id", "formula", "crystal_system", "density", "volume", 
        "e_above_hull", "formation_energy", "material_class", "is_viable"
    ]]
    materials_table.to_sql("materials", conn, if_exists="replace", index=False)

    electronic_table = df[[
        "mp_id", "band_gap_dft", "delta_eg_predicted", "band_gap_calibrated",
        "eps_ehp_ev", "Voc_est_v", "theoretical_efficiency_pct"
    ]]
    electronic_table.to_sql("electronic_properties", conn, if_exists="replace", index=False)

    perf_cols = [
        "mp_id", "ed_est_ev", "radiation_resistance_score",
        "carriers_per_electron_Ni63", "penetration_depth_um_Ni63",
        "carriers_per_electron_H3", "penetration_depth_um_H3",
        "carriers_per_electron_C14", "penetration_depth_um_C14",
        "carriers_per_electron_Pm147", "penetration_depth_um_Pm147"
    ]
    df[perf_cols].to_sql("betavoltaic_performance", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

    print(f"\n Реляционная база данных успешно сохранена в: {db_path}")
    print(f" Записано записей: {len(df)} в 3 связанные таблицы!")
    print(f" Жизнеспособных полупроводников: {df['is_viable'].sum()} из {len(df)}")


if __name__ == "__main__":
    build_database()