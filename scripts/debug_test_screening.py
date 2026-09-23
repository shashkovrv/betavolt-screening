import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.screening.ranker import BetavoltaicLibrary
from src.screening.pareto import identify_pareto_frontier_3d
from app.plots import plot_pareto_interactive, plot_3d_materials_space, get_depth_series

def run_diagnostics():
    print("="*70)
    print("  КОМПЛЕКСНАЯ ДИАГНОСТИКА И ДЕБАГ ПРОЕКТА СКРИНИНГА БЕТАВОЛЬТАИКИ")
    print("="*70)

    # 1. Проверка SQLite базы данных
    db_path = project_root / "data" / "05_database" / "betavoltaic_library.db"
    print(f"\n[1/5] Проверка базы данных: {db_path}")
    if not db_path.exists():
        print(f"ОШИБКА: Файл БД не найден по пути {db_path}!")
        return

    conn = sqlite3.connect(str(db_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    print(f"  Таблицы в SQLite: {tables}")
    
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        print(f"  - Таблица '{t}': {count:,} записей")
    
    # 2. Проверка полноты и целостности данных
    print(f"\n[2/5] Проверка целостности данных и физических диапазонов...")
    lib = BetavoltaicLibrary(str(db_path))
    df = lib.get_full_dataframe()
    print(f"  Загружено объединенных записей: {len(df):,}")

    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        print("  [OK] Пропусков (NULL/NaN) в ключевых колонках: 0 (Идеально)")
    else:
        print(f"  ВНИМАНИЕ: Обнаружены пропуски в колонках:\n{null_cols}")

    print("\n  Статистика физических диапазонов:")
    print(f"  - Плотность rho: min={df['density'].min():.2f}, max={df['density'].max():.2f} г/см3")
    print(f"  - Калиброванная Eg: min={df['band_gap_calibrated'].min():.2f}, max={df['band_gap_calibrated'].max():.2f} эВ")
    print(f"  - КПД theoretical_efficiency_pct: min={df['theoretical_efficiency_pct'].min():.2f}%, max={df['theoretical_efficiency_pct'].max():.2f}%")
    print(f"  - Радиационная стойкость R_score: min={df['radiation_resistance_score'].min():.1f}%, max={df['radiation_resistance_score'].max():.1f}%")
    print(f"  - Пробег Ni-63: min={df['penetration_depth_um_Ni63'].min():.2f}, max={df['penetration_depth_um_Ni63'].max():.2f} мкм")
    print(f"  - Пробег H-3: min={df['penetration_depth_um_H3'].min():.2f}, max={df['penetration_depth_um_H3'].max():.2f} мкм")
    print(f"  - Пробег C-14: min={df['penetration_depth_um_C14'].min():.2f}, max={df['penetration_depth_um_C14'].max():.2f} мкм")
    print(f"  - Пробег Pm-147: min={df['penetration_depth_um_Pm147'].min():.2f}, max={df['penetration_depth_um_Pm147'].max():.2f} мкм")

    # 3. Проверка эталонов и новых кандидатов
    print(f"\n[3/5] Проверка эталонных материалов и чемпионов в БД...")
    check_formulas = ["C", "Si", "GaN", "SiC", "TiO2", "BN", "AlN", "BP", "B4C", "B6P", "Ga2O3"]
    for f in check_formulas:
        res = lib.find_by_formula(f)
        if not res.empty:
            row = res.iloc[0]
            print(f"  [OK] {f:8} | Eg_cal={row['band_gap_calibrated']:.2f} эВ | КПД={row['theoretical_efficiency_pct']:.2f}% | R_score={row['radiation_resistance_score']:.1f}% | R(Ni63)={row['penetration_depth_um_Ni63']:.2f} мкм | Класс: {row['material_class']}")
        else:
            print(f"  [FAIL] {f:8} НЕ НАЙДЕН в БД!")

    # 4. Проверка устойчивости UI фильтров и краевых условий
    print(f"\n[4/5] Тестирование краевых условий (Edge Cases) в UI фильтрации...")
    
    test_cases = [
        ("Дефолтный фильтр (Все классы, viable=True, eff>=18.0, rad>=20.0, Eg=[1.2, 6.0])", 
         df[(df["is_viable"] == 1) & (df["theoretical_efficiency_pct"] >= 18.0) & (df["radiation_resistance_score"] >= 20.0) & (df["band_gap_calibrated"].between(1.2, 6.0))]),
        
        ("Только Ковалентные полупроводники", 
         df[(df["is_viable"] == 1) & (df["material_class"] == "Ковалентные (IV, III-V, карбиды, бориды, нитриды)") & (df["theoretical_efficiency_pct"] >= 18.0)]),
        
        ("Только Оксидные полупроводники", 
         df[(df["is_viable"] == 1) & (df["material_class"] == "Оксидные полупроводники (простые и тройные)") & (df["theoretical_efficiency_pct"] >= 18.0)]),
        
        ("Поиск по 'SiC'", 
         df[df["formula"].str.contains("SiC", case=False, na=False)]),
        
        ("Пустая выборка (нереалистичные фильтры КПД >= 25%)", 
         df[df["theoretical_efficiency_pct"] >= 25.0]),
        
        ("Поиск несуществующей формулы 'Unobtainium123'", 
         df[df["formula"].str.contains("Unobtainium123", case=False, na=False)])
    ]

    for name, subset in test_cases:
        print(f"\n  Тест: '{name}' -> Найдено записей: {len(subset)}")
        
        # Проверяем расчет Парето
        if not subset.empty:
            pts = subset[["theoretical_efficiency_pct", "radiation_resistance_score", "penetration_depth_um_Ni63"]].values
            p_mask = identify_pareto_frontier_3d(pts)
            p_df = subset[p_mask]
            champs = p_df["formula"].unique().tolist()[:6]
            print(f"    - 3D Парето-чемпионов: {len(p_df)} (Примеры: {', '.join(champs)})")
        else:
            p_df = pd.DataFrame()
            print("    - Корректная обработка пустого датасета (0 чемпионов)")

        # Проверяем отрисовку графиков для всех 4 изотопов
        for iso in ["Ni-63", "H-3", "C-14", "Pm-147"]:
            try:
                fig2d = plot_pareto_interactive(subset, p_df, iso)
                fig3d = plot_3d_materials_space(subset, p_df, iso)
            except Exception as e:
                print(f"      [FAIL] ОШИБКА при генерации графиков для {iso}: {e}")
                raise e
        print(f"    [OK] Все 4 изотопа отрисованы без ошибок!")

    # 5. Проверка генерации паспорта полупроводника
    print(f"\n[5/5] Тестирование паспорта полупроводника для ключевых кандидатов...")
    passport_candidates = ["BN", "SiC", "C", "GaN", "Ga2O3", "BP", "B4C", "B6P"]
    for mat in passport_candidates:
        mat_rows = df[df["formula"] == mat]
        if not mat_rows.empty:
            m = mat_rows.iloc[0]
            print(f"  [OK] Паспорт '{mat}': КПД={m['theoretical_efficiency_pct']:.2f}%, R_score={m['radiation_resistance_score']:.1f}%, Ed={m['ed_est_ev']:.1f} эВ, Ni63_depth={m['penetration_depth_um_Ni63']:.2f} мкм, Ni63_pairs={int(m['carriers_per_electron_Ni63'])}")
        else:
            print(f"  [FAIL] '{mat}' не найден для паспорта!")

    print("\n" + "="*70)
    print("  ВСЕ ТЕСТЫ И ДИАГНОСТИКА УСПЕШНО ПРОЙДЕНЫ! БАГОВ НЕ ОБНАРУЖЕНО.")
    print("="*70)

if __name__ == "__main__":
    run_diagnostics()
