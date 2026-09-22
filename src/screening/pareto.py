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


def identify_pareto_frontier_2d(costs: np.ndarray) -> np.ndarray:
    """
    Находит индексы Парето-оптимальных точек (максимизация обеих осей).
    costs: массив формы (N, 2), где столбцы [КПД, Стойкость].
    """
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # Точка i доминируется точкой j, если точка j строго больше по обеим координатам
            is_pareto[is_pareto] = ~(
                (costs[is_pareto, 0] <= c[0]) & 
                (costs[is_pareto, 1] <= c[1]) & 
                ((costs[is_pareto, 0] < c[0]) | (costs[is_pareto, 1] < c[1]))
            )
            is_pareto[i] = True
    return is_pareto


def run_pareto_screening(
    output_fig_path: str = "reports/figures/pareto_frontier.html",
    output_csv_path: str = "reports/figures/pareto_champions.csv"
):
    print(" Запуск многокритериального Парето-скрининга...")

    # 1. Читаем полную библиотеку через наш API
    lib = BetavoltaicLibrary()
    df = lib.get_full_dataframe()
    print(f"  Загружено материалов из SQLite: {len(df)}")

    # 2. Первичный отбор по термодинамической стабильности (синтезируемость)
    stable_df = df[df["e_above_hull"] <= 0.01].copy().reset_index(drop=True)
    print(f"  Строго стабильных кандидатов (Ehull <= 0.01 эВ): {len(stable_df)}")

    # 3. Вычисляем Парето-фронт по осям [КПД, Радиационная стойкость]
    points = stable_df[["theoretical_efficiency_pct", "radiation_resistance_score"]].values
    pareto_mask = identify_pareto_frontier_2d(points)
    
    stable_df["is_pareto"] = pareto_mask
    pareto_df = stable_df[pareto_mask].sort_values(by="theoretical_efficiency_pct", ascending=False)
    
    print(f"   Выделено Парето-оптимальных чемпионов: {len(pareto_df)} материалов!")

    # 4. Сохраняем список чемпионов
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    cols_to_save = [
        "formula", "mp_id", "crystal_system", "density", 
        "band_gap_calibrated", "theoretical_efficiency_pct", 
        "ed_est_ev", "radiation_resistance_score"
    ]
    pareto_df[cols_to_save].to_csv(output_csv_path, index=False)
    print(f"  Таблица чемпионов сохранена в: {output_csv_path}")

    # 5. Строим график Парето-фронта
    print("  Отрисовка графика Парето-фронта для диплома...")
    
    # Не-Парето точки (серые)
    fig = px.scatter(
        stable_df[~stable_df["is_pareto"]],
        x="theoretical_efficiency_pct",
        y="radiation_resistance_score",
        color="band_gap_calibrated",
        hover_name="formula",
        hover_data=["mp_id", "crystal_system", "density"],
        opacity=0.4,
        labels={
            "theoretical_efficiency_pct": "Теоретический КПД, %",
            "radiation_resistance_score": "Индекс радиационной стойкости (0-100)",
            "band_gap_calibrated": "Eg калибр. (эВ)"
        },
        title="Многокритериальный Парето-скрининг бетавольтаических полупроводников"
    )

    # Парето-чемпионы (яркие красные звезды)
    fig.add_trace(
        go.Scatter(
            x=pareto_df["theoretical_efficiency_pct"],
            y=pareto_df["radiation_resistance_score"],
            mode="markers+text",
            text=pareto_df["formula"],
            textposition="top center",
            marker=dict(size=12, color="red", symbol="star"),
            name="Парето-фронт (Чемпионы)"
        )
    )

    fig.update_layout(template="plotly_white", width=1000, height=600)
    
    # Сохраняем интерактивный HTML и статическую картинку
    fig.write_html(output_fig_path)
    print(f" График успешно сохранен в: {output_fig_path}")

    # Выводим топ-5 Парето чемпионов
    print("\n ТОП-5 ПАРЕТО-ОПТИМАЛЬНЫХ МАТЕРИАЛОВ (Глава 4 диплома):")
    print(pareto_df[cols_to_save].head(5).to_string(index=False))

    return pareto_df


if __name__ == "__main__":
    run_pareto_screening()