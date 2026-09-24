import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.physics.betavoltaics import calculate_penetration_depth

CRYSTAL_SYSTEMS_RU = {
    "cubic": "Кубическая",
    "hexagonal": "Гексагональная",
    "tetragonal": "Тетрагональная",
    "orthorhombic": "Ромбическая",
    "trigonal": "Тригональная",
    "monoclinic": "Моноклинная",
    "triclinic": "Триклинная"
}


def get_depth_series(dataframe: pd.DataFrame, isotope: str) -> pd.Series:
    """Безопасно возвращает серию глубин пробега для заданного изотопа."""
    clean_tag = isotope.replace("-", "")
    depth_col = f"penetration_depth_um_{clean_tag}"
    if depth_col in dataframe.columns:
        return dataframe[depth_col].round(2)
    elif "density" in dataframe.columns:
        return dataframe["density"].apply(lambda rho: calculate_penetration_depth(rho, isotope)).round(2)
    return pd.Series(0.0, index=dataframe.index)


def plot_pareto_interactive(df: pd.DataFrame, pareto_df: pd.DataFrame, isotope: str):
    """Строит интерактивный 2D график Парето-скрининга на русском языке в строгом академическом стиле."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=580,
            xaxis=dict(title="Теоретический КПД преобразования η (%)", showgrid=True),
            yaxis=dict(title="Индекс радиационной стойкости R_score (отн. алмаза, %)", showgrid=True),
            annotations=[
                dict(
                    text="Нет данных, удовлетворяющих заданным критериям фильтрации",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
            ]
        )
        return fig

    plot_df = df.copy()
    plot_df["Сингония"] = plot_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(plot_df["crystal_system"])
    plot_df["КПД (%)"] = pd.to_numeric(plot_df["theoretical_efficiency_pct"], errors="coerce").round(2)
    plot_df["Стойкость"] = pd.to_numeric(plot_df["radiation_resistance_score"], errors="coerce").round(1)
    plot_df["Eg (эВ)"] = pd.to_numeric(plot_df["band_gap_calibrated"], errors="coerce").round(2)
    plot_df["Глубина пробега (мкм)"] = get_depth_series(plot_df, isotope)
    plot_df["formula_clean"] = plot_df["formula"].fillna(plot_df["mp_id"])

    # Явно упаковываем custom_data: [Eg, Сингония, Глубина пробега, ID]
    custom_cols = ["Eg (эВ)", "Сингония", "Глубина пробега (мкм)", "mp_id"]

    fig = px.scatter(
        plot_df,
        x="КПД (%)",
        y="Стойкость",
        color="Eg (эВ)",
        hover_name="formula_clean",
        custom_data=custom_cols,
        opacity=0.60,
        color_continuous_scale="Viridis",
        labels={
            "КПД (%)": "Теоретический КПД преобразования η (%)",
            "Стойкость": "Индекс радиационной стойкости R_score (отн. алмаза, %)",
            "Eg (эВ)": "Eg калибр. (эВ)"
        },
        title=f"Многокритериальный 3D Парето-скрининг материалов (Изотоп: {isotope})"
    )

    # Строгий академический тултип для фонового облака точек
    fig.update_traces(
        selector=dict(type="scatter"),
        hovertemplate=(
            "<b>Материал: %{hovertext}</b><br>"
            "Теор. КПД: %{x:.2f}%<br>"
            "Индекс стойкости: %{y:.1f}%<br>"
            "Eg калибр.: %{customdata[0]:.2f} эВ<br>"
            "Сингония: %{customdata[1]}<br>"
            f"Глубина пробега ({isotope}): " + "%{customdata[2]:.2f} мкм<br>"
            "ID Materials Project: %{customdata[3]}<extra></extra>"
        )
    )

    # Добавляем 3D Парето-чемпионов отдельным акцентным слоем
    if pareto_df is not None and not pareto_df.empty:
        p_df = pareto_df.copy()
        p_df["Сингония"] = p_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(p_df["crystal_system"])
        p_df["Глубина пробега (мкм)"] = get_depth_series(p_df, isotope)
        p_df["Eg (эВ)"] = pd.to_numeric(p_df["band_gap_calibrated"], errors="coerce").round(2)
        p_df["formula_clean"] = p_df["formula"].fillna(p_df["mp_id"])
        
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(p_df["theoretical_efficiency_pct"], errors="coerce").round(2),
                y=pd.to_numeric(p_df["radiation_resistance_score"], errors="coerce").round(1),
                mode="markers+text",
                text=p_df["formula_clean"],
                textposition="top center",
                textfont=dict(size=12, color="#B22222", family="Arial Black"),
                marker=dict(
                    size=14, 
                    color="#FF2A2A", 
                    symbol="star",
                    line=dict(width=1.5, color="#500000")
                ),
                name="3D Парето-чемпионы (кликните для фильтра)",
                customdata=p_df[["formula_clean", "Eg (эВ)", "Сингония", "Глубина пробега (мкм)", "mp_id"]].values,
                hovertemplate=(
                    "<b>⭐ 3D Парето-чемпион: %{customdata[0]}</b><br>"
                    "Теор. КПД: %{x:.2f}%<br>"
                    "Индекс стойкости: %{y:.1f}%<br>"
                    "Eg калибр.: %{customdata[1]:.2f} эВ<br>"
                    "Сингония: %{customdata[2]}<br>"
                    f"Глубина пробега ({isotope}): " + "%{customdata[3]:.2f} мкм<br>"
                    "ID Materials Project: %{customdata[4]}<extra></extra>"
                )
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
        xaxis=dict(
            title="Теоретический КПД преобразования η (%)",
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False
        ),
        yaxis=dict(
            title="Индекс радиационной стойкости R_score (отн. алмаза, %)",
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False
        ),
        coloraxis_colorbar=dict(
            title="Eg калибр. (эВ)",
            thickness=16,
            len=0.85
        ),
        margin=dict(l=50, r=40, t=75, b=45)
    )
    return fig


def plot_3d_materials_space(df: pd.DataFrame, pareto_df: pd.DataFrame = None, isotope: str = "Ni-63", sample_size: int = 2500):
    """Строит трехмерное пространство критериев Парето-оптимизации (КПД, Стойкость, Пробег)."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=680,
            scene=dict(
                xaxis_title="Теор. КПД η (%) [max]",
                yaxis_title="Стойкость R_score (%) [max]",
                zaxis_title=f"Пробег {isotope} R (мкм) [min]"
            ),
            annotations=[
                dict(
                    text="Нет данных для 3D отображения",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
            ]
        )
        return fig

    plot_df = df.sample(min(len(df), sample_size), random_state=42).copy()
    plot_df["Сингония"] = plot_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(plot_df["crystal_system"])
    plot_df["КПД (%)"] = pd.to_numeric(plot_df["theoretical_efficiency_pct"], errors="coerce").round(2)
    plot_df["Стойкость"] = pd.to_numeric(plot_df["radiation_resistance_score"], errors="coerce").round(1)
    plot_df["Пробег (мкм)"] = get_depth_series(plot_df, isotope)
    plot_df["Eg (эВ)"] = pd.to_numeric(plot_df["band_gap_calibrated"], errors="coerce").round(2)
    plot_df["formula_clean"] = plot_df["formula"].fillna(plot_df["mp_id"])

    color_col = "material_class" if "material_class" in plot_df.columns else "Сингония"
    custom_cols_3d = ["Eg (эВ)", "Сингония", "mp_id"]

    fig = px.scatter_3d(
        plot_df,
        x="КПД (%)",
        y="Стойкость",
        z="Пробег (мкм)",
        color=color_col,
        hover_name="formula_clean",
        custom_data=custom_cols_3d,
        opacity=0.65,
        labels={
            "КПД (%)": "Теор. КПД η (%) [max]",
            "Стойкость": "Стойкость R_score (%) [max]",
            "Пробег (мкм)": f"Пробег {isotope} R (мкм) [min]",
            "material_class": "Класс материала"
        },
        title=f"3D Пространство Парето-критериев ({isotope}): КПД (max) — Стойкость (max) — Пробег (min)"
    )

    # Настройка тултипа для фоновых точек 3D
    fig.update_traces(
        selector=dict(type="scatter3d"),
        hovertemplate=(
            "<b>Материал: %{hovertext}</b><br>"
            "Теор. КПД: %{x:.2f}%<br>"
            "Индекс стойкости: %{y:.1f}%<br>"
            f"Пробег {isotope}: " + "%{z:.2f} мкм<br>"
            "Eg калибр.: %{customdata[0]:.2f} эВ<br>"
            "Сингония: %{customdata[1]}<br>"
            "ID: %{customdata[2]}<extra></extra>"
        )
    )

    # Наложение 3D Парето-чемпионов
    if pareto_df is not None and not pareto_df.empty:
        p_df = pareto_df.copy()
        p_df["Сингония"] = p_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(p_df["crystal_system"])
        p_df["Eg (эВ)"] = pd.to_numeric(p_df["band_gap_calibrated"], errors="coerce").round(2)
        p_df["formula_clean"] = p_df["formula"].fillna(p_df["mp_id"])
        p_depth = get_depth_series(p_df, isotope)
        
        fig.add_trace(
            go.Scatter3d(
                x=pd.to_numeric(p_df["theoretical_efficiency_pct"], errors="coerce").round(2),
                y=pd.to_numeric(p_df["radiation_resistance_score"], errors="coerce").round(1),
                z=p_depth,
                mode="markers+text",
                text=p_df["formula_clean"],
                textposition="top center",
                textfont=dict(size=11, color="red"),
                marker=dict(size=7, color="#E60000", symbol="diamond", line=dict(width=1, color="darkred")),
                name="3D Парето-чемпионы",
                customdata=p_df[["formula_clean", "Eg (эВ)", "Сингония", "mp_id"]].values,
                hovertemplate=(
                    "<b>⭐ 3D Парето-чемпион: %{customdata[0]}</b><br>"
                    "Теор. КПД: %{x:.2f}%<br>"
                    "Индекс стойкости: %{y:.1f}%<br>"
                    f"Пробег {isotope}: " + "%{z:.2f} мкм<br>"
                    "Eg калибр.: %{customdata[1]:.2f} эВ<br>"
                    "Сингония: %{customdata[2]}<br>"
                    "ID: %{customdata[3]}<extra></extra>"
                )
            )
        )

    fig.update_layout(
        height=680,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            title="Классификация:",
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02
        ),
        scene=dict(
            xaxis=dict(title="Теор. КПД η (%) [max]", backgroundcolor="rgba(245,247,250,0.5)"),
            yaxis=dict(title="Стойкость R_score (%) [max]", backgroundcolor="rgba(245,247,250,0.5)"),
            zaxis=dict(title=f"Пробег {isotope} R (мкм) [min]", backgroundcolor="rgba(245,247,250,0.5)")
        )
    )
    return fig