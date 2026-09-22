import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_pareto_interactive(df: pd.DataFrame, pareto_df: pd.DataFrame, isotope: str):
    """Строит интерактивный 2D график Парето-скрининга."""
    clean_tag = isotope.replace("-", "")
    
    fig = px.scatter(
        df,
        x="theoretical_efficiency_pct",
        y="radiation_resistance_score",
        color="band_gap_calibrated",
        hover_name="formula",
        hover_data=["mp_id", "crystal_system", f"penetration_depth_um_{clean_tag}"],
        opacity=0.45,
        labels={
            "theoretical_efficiency_pct": "Теоретический КПД (%)",
            "radiation_resistance_score": "Индекс радиационной стойкости (0-100)",
            "band_gap_calibrated": "Eg калибр. (эВ)"
        },
        title=f"Многокритериальный скрининг под изотоп {isotope}"
    )

    # Добавляем Парето-чемпионов
    fig.add_trace(
        go.Scatter(
            x=pareto_df["theoretical_efficiency_pct"],
            y=pareto_df["radiation_resistance_score"],
            mode="markers+text",
            text=pareto_df["formula"],
            textposition="top center",
            marker=dict(size=13, color="red", symbol="star"),
            name="Парето-чемпионы"
        )
    )
    fig.update_layout(template="plotly_white", height=550)
    return fig


def plot_3d_materials_space(df: pd.DataFrame, sample_size: int = 2500):
    """Строит трехмерное пространство свойств: Зона x КПД x Радиационная стойкость."""
    plot_df = df.sample(min(len(df), sample_size), random_state=42)
    
    fig = px.scatter_3d(
        plot_df,
        x="band_gap_calibrated",
        y="theoretical_efficiency_pct",
        z="radiation_resistance_score",
        color="crystal_system",
        hover_name="formula",
        opacity=0.7,
        labels={
            "band_gap_calibrated": "Eg калибр (эВ)",
            "theoretical_efficiency_pct": "КПД (%)",
            "radiation_resistance_score": "Стойкость (0-100)"
        },
        title="3D-пространство физических характеристик библиотеки"
    )
    fig.update_layout(height=650)
    return fig