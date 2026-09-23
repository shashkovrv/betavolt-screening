import sys
from pathlib import Path

# Защита путей для кнопки Play
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.screening.ranker import BetavoltaicLibrary


def identify_pareto_frontier_3d(points: np.ndarray) -> np.ndarray:
    """
    Находит индексы 3D Парето-оптимальных точек:
    points: массив формы (N, 3), где столбцы:
      [0: КПД (максимизация), 1: Радиационная стойкость (максимизация), 2: Глубина пробега R (минимизация)].
    """
    norm = points.copy()
    norm[:, 2] = -norm[:, 2]  # Минимизация R -> максимизация -R
    is_pareto = np.ones(norm.shape[0], dtype=bool)
    
    for i, c in enumerate(norm):
        if is_pareto[i]:
            # Точка i доминируется точкой j, если j >= i по всем осям и строго больше хотя бы по одной
            is_pareto[is_pareto] = ~(
                (norm[is_pareto, 0] <= c[0]) & 
                (norm[is_pareto, 1] <= c[1]) & 
                (norm[is_pareto, 2] <= c[2]) & 
                ((norm[is_pareto, 0] < c[0]) | (norm[is_pareto, 1] < c[1]) | (norm[is_pareto, 2] < c[2]))
            )
            is_pareto[i] = True
    return is_pareto


def run_pareto_screening(
    output_fig_path: str = "reports/figures/pareto_frontier.html",
    output_csv_path: str = "reports/figures/pareto_champions.csv"
):
    print("  Запуск трехкритериального (3D) Парето-скрининга...")

    # 1. Читаем полную библиотеку через наш API
    lib = BetavoltaicLibrary()
    df = lib.get_full_dataframe()
    print(f"  Загружено материалов из SQLite: {len(df)}")

    # 2. Первичный отбор по термодинамической стабильности и жизнеспособности полупроводников
    viable_df = df[
        (df["is_viable"] == 1) & 
        (df["e_above_hull"] <= 0.01) & 
        (df["band_gap_calibrated"].between(1.0, 7.5))
    ].copy().reset_index(drop=True)
    print(f"  Стабильных жизнеспособных полупроводников (Ehull <= 0.01 эВ, Eg 1.0-7.5 эВ): {len(viable_df)}")

    # 3. Вычисляем 3D Парето-фронт по осям [КПД (max), Радиационная стойкость (max), Пробег Ni-63 (min)]
    points = viable_df[["theoretical_efficiency_pct", "radiation_resistance_score", "penetration_depth_um_Ni63"]].values
    pareto_mask = identify_pareto_frontier_3d(points)
    
    viable_df["is_pareto"] = pareto_mask
    pareto_df = viable_df[pareto_mask].sort_values(by="theoretical_efficiency_pct", ascending=False)
    
    print(f"  Выделено 3D Парето-оптимальных чемпионов: {len(pareto_df)} материалов!")

    # 4. Сохраняем список чемпионов
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cols_to_save = [
        "formula", "mp_id", "crystal_system", "material_class", "density", 
        "band_gap_calibrated", "theoretical_efficiency_pct", 
        "ed_est_ev", "radiation_resistance_score",
        "penetration_depth_um_Ni63", "carriers_per_electron_Ni63",
        "penetration_depth_um_H3", "carriers_per_electron_H3",
        "penetration_depth_um_C14", "carriers_per_electron_C14",
        "penetration_depth_um_Pm147", "carriers_per_electron_Pm147"
    ]
    cols_to_save = [c for c in cols_to_save if c in pareto_df.columns]
    pareto_df[cols_to_save].to_csv(output_csv_path, index=False)
    print(f"  Таблица 3D чемпионов сохранена в: {output_csv_path}")

    # 5. Строим график Парето-фронта
    print("  Отрисовка интерактивного графика 3D Парето-скрининга...")
    
    fig = px.scatter(
        viable_df[~viable_df["is_pareto"]],
        x="theoretical_efficiency_pct",
        y="radiation_resistance_score",
        color="band_gap_calibrated",
        size="density",
        hover_name="formula",
        hover_data=["mp_id", "crystal_system", "material_class", "density", "penetration_depth_um_Ni63"],
        opacity=0.45,
        labels={
            "theoretical_efficiency_pct": "Теоретический КПД, %",
            "radiation_resistance_score": "Индекс радиационной стойкости (0-100)",
            "band_gap_calibrated": "Eg калибр. (эВ)",
            "density": "Плотность (г/см³)"
        },
        title="Многокритериальный 3D Парето-скрининг бетавольтаических полупроводников (Ni-63)"
    )

    # Парето-чемпионы (яркие красные звезды)
    fig.add_trace(
        go.Scatter(
            x=pareto_df["theoretical_efficiency_pct"],
            y=pareto_df["radiation_resistance_score"],
            mode="markers+text",
            text=pareto_df["formula"],
            textposition="top center",
            marker=dict(size=13, color="red", symbol="star"),
            name="3D Парето-чемпионы"
        )
    )

    fig.update_layout(template="plotly_white", width=1000, height=600)
    
    # Сохраняем интерактивный HTML
    fig.write_html(output_fig_path)
    print(f"  График успешно сохранен в: {output_fig_path}")

    # Выводим топ Парето чемпионов
    print("\n ТОП 3D ПАРЕТО-ОПТИМАЛЬНЫХ МАТЕРИАЛОВ:")
    print(pareto_df[cols_to_save].head(10).to_string(index=False))

    return pareto_df


if __name__ == "__main__":
    run_pareto_screening()