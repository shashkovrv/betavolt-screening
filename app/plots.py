import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

CRYSTAL_SYSTEMS_RU = {
    "cubic": "Кубическая",
    "hexagonal": "Гексагональная",
    "tetragonal": "Тетрагональная",
    "orthorhombic": "Ромбическая",
    "trigonal": "Тригональная",
    "monoclinic": "Моноклинная",
    "triclinic": "Триклинная"
}


def plot_pareto_interactive(df: pd.DataFrame, pareto_df: pd.DataFrame, isotope: str):
    """Строит интерактивный 2D график Парето-скрининга на русском языке."""
    clean_tag = isotope.replace("-", "")
    depth_col = f"penetration_depth_um_{clean_tag}"
    
    # Подготавливаем отображаемые данные
    plot_df = df.copy()
    plot_df["Сингония"] = plot_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(plot_df["crystal_system"])
    plot_df["КПД (%)"] = plot_df["theoretical_efficiency_pct"].round(2)
    plot_df["Стойкость"] = plot_df["radiation_resistance_score"].round(1)
    plot_df["Eg (эВ)"] = plot_df["band_gap_calibrated"].round(2)
    plot_df["Глубина пробега (мкм)"] = plot_df[depth_col].round(2) if depth_col in plot_df.columns else 0.0

    fig = px.scatter(
        plot_df,
        x="КПД (%)",
        y="Стойкость",
        color="Eg (эВ)",
        hover_name="formula",
        hover_data={
            "КПД (%)": True,
            "Стойкость": True,
            "Eg (эВ)": True,
            "Сингония": True,
            "Глубина пробега (мкм)": True,
            "mp_id": True
        },
        opacity=0.55,
        color_continuous_scale="Viridis",
        labels={
            "КПД (%)": "Теоретический КПД преобразования (%)",
            "Стойкость": "Радиационная стойкость R_score (отн. алмаза, %)",
            "Eg (эВ)": "Eg калибр. (эВ)"
        },
        title=f"Многокритериальный 3D Парето-скрининг материалов (Изотоп: {isotope})"
    )

    # Настройка тултипа
    fig.update_traces(
        hovertemplate="<b>Материал: %{hovertext}</b><br>" +
                      "Теор. КПД: %{x:.2f}%<br>" +
                      "Индекс стойкости: %{y:.1f}%<br>" +
                      "Eg калибр.: %{customdata[2]:.2f} эВ<br>" +
                      "Сингония: %{customdata[3]}<br>" +
                      "Глубина пробега: %{customdata[4]:.2f} мкм<br>" +
                      "ID: %{customdata[5]}<extra></extra>"
    )

    # Добавляем Парето-чемпионов
    if not pareto_df.empty:
        p_df = pareto_df.copy()
        p_df["Сингония"] = p_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(p_df["crystal_system"])
        p_df["Глубина пробега (мкм)"] = p_df[depth_col].round(2) if depth_col in p_df.columns else 0.0
        
        fig.add_trace(
            go.Scatter(
                x=p_df["theoretical_efficiency_pct"].round(2),
                y=p_df["radiation_resistance_score"].round(1),
                mode="markers+text",
                text=p_df["formula"],
                textposition="top center",
                textfont=dict(size=12, color="#B22222", family="Arial Black"),
                marker=dict(
                    size=14, 
                    color="#FF2A2A", 
                    symbol="star",
                    line=dict(width=1.5, color="#500000")
                ),
                name="3D Парето-чемпионы (кликните, чтобы скрыть)",
                customdata=p_df[["formula", "crystal_system", "band_gap_calibrated", "mp_id", "Сингония", "Глубина пробега (мкм)"]],
                hovertemplate="<b>3D Парето-чемпион: %{customdata[0]}</b><br>" +
                              "Теор. КПД: %{x:.2f}%<br>" +
                              "Индекс стойкости: %{y:.1f}%<br>" +
                              "Eg калибр.: %{customdata[2]:.2f} эВ<br>" +
                              "Сингония: %{customdata[4]}<br>" +
                              "Глубина пробега: %{customdata[5]:.2f} мкм<br>" +
                              "ID: %{customdata[3]}<extra></extra>"
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=580,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            title=None
        ),
        margin=dict(l=40, r=40, t=70, b=40)
    )
    return fig


def plot_3d_materials_space(df: pd.DataFrame, pareto_df: pd.DataFrame = None, isotope: str = "Ni-63", sample_size: int = 2500):
    """Строит трехмерное пространство критериев Парето-оптимизации (КПД, Стойкость, Пробег)."""
    clean_tag = isotope.replace("-", "")
    depth_col = f"penetration_depth_um_{clean_tag}"
    
    plot_df = df.sample(min(len(df), sample_size), random_state=42).copy()
    plot_df["Сингония"] = plot_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(plot_df["crystal_system"])
    plot_df["КПД (%)"] = plot_df["theoretical_efficiency_pct"].round(2)
    plot_df["Стойкость"] = plot_df["radiation_resistance_score"].round(1)
    plot_df["Пробег (мкм)"] = plot_df[depth_col].round(2) if depth_col in plot_df.columns else 0.0

    color_col = "material_class" if "material_class" in plot_df.columns else "Сингония"

    fig = px.scatter_3d(
        plot_df,
        x="КПД (%)",
        y="Стойкость",
        z="Пробег (мкм)",
        color=color_col,
        hover_name="formula",
        opacity=0.65,
        labels={
            "КПД (%)": "Теор. КПД (%) [max]",
            "Стойкость": "Стойкость R_score (%) [max]",
            "Пробег (мкм)": f"Пробег {isotope} (мкм) [min]",
            "material_class": "Класс материала"
        },
        title=f"3D Пространство Парето-критериев ({isotope}): КПД (max) — Стойкость (max) — Пробег (min)"
    )

    # Наложение 3D Парето-чемпионов
    if pareto_df is not None and not pareto_df.empty:
        p_df = pareto_df.copy()
        p_depth = p_df[depth_col].round(2) if depth_col in p_df.columns else 0.0
        fig.add_trace(
            go.Scatter3d(
                x=p_df["theoretical_efficiency_pct"].round(2),
                y=p_df["radiation_resistance_score"].round(1),
                z=p_depth,
                mode="markers+text",
                text=p_df["formula"],
                textposition="top center",
                textfont=dict(size=11, color="red"),
                marker=dict(size=7, color="red", symbol="diamond"),
                name="3D Парето-чемпионы"
            )
        )

    fig.update_layout(
        height=680,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(title="Классификация:")
    )
    return fig