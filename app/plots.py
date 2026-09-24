import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.physics.betavoltaics import calculate_penetration_depth
from src.physics.radiation_transport import (
    compute_normalized_fermi_spectrum,
    joy_luo_stopping_power_exact,
    solve_energy_loss_ode,
    mckinley_feshbach_displacement_cross_section,
    compute_spectrum_averaged_displacement_cross_section,
    ISOTOPE_DECAY_PARAMS,
    M_E_C2_KEV,
    M_U_C2_KEV,
)

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


def plot_nist_golden_benchmark(b_df: pd.DataFrame) -> go.Figure:
    """Интерактивный двухпанельный график сверки с эталонами NIST/Landolt-Börnstein."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Сопоставление ширины запрещенной зоны Eg (эВ)</b>",
            "<b>Снижение абсолютной погрешности |ΔEg| (эВ)</b>"
        ],
        horizontal_spacing=0.12
    )

    formulas = b_df["Формула"].tolist()
    dft_vals = b_df["DFT PBE (эВ)"].tolist()
    ml_vals = b_df["Калибр. Eg (эВ)"].tolist()
    exp_vals = b_df["Эксперимент (эВ)"].tolist()
    dft_errs = b_df["Ошибка DFT (эВ)"].tolist()
    ml_errs = b_df["Ошибка ML (эВ)"].tolist()
    sources = b_df["Первоисточник"].tolist()

    # Левая панель: Значения Eg
    fig.add_trace(
        go.Bar(
            name="DFT PBE (Исходный)",
            x=formulas,
            y=dft_vals,
            marker_color="#D9534F",
            customdata=sources,
            hovertemplate="<b>%{x}</b><br>DFT PBE: %{y:.2f} эВ<br>Источник: %{customdata}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name="Δ-ML CatBoost (Калибр.)",
            x=formulas,
            y=ml_vals,
            marker_color="#0275D8",
            customdata=sources,
            hovertemplate="<b>%{x}</b><br>Δ-ML: %{y:.2f} эВ<br>Источник: %{customdata}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            name="Эксперимент (NIST/LB)",
            x=formulas,
            y=exp_vals,
            marker_color="#2E7D32",
            customdata=sources,
            hovertemplate="<b>%{x}</b><br>Эксперимент: %{y:.2f} эВ<br>Источник: %{customdata}<extra></extra>"
        ),
        row=1, col=1
    )

    # Правая панель: Ошибки
    fig.add_trace(
        go.Bar(
            name="Ошибка DFT PBE",
            x=formulas,
            y=dft_errs,
            marker_color="#E57373",
            showlegend=False,
            customdata=sources,
            hovertemplate="<b>%{x}</b><br>Ошибка DFT: %{y:.2f} эВ<br>Источник: %{customdata}<extra></extra>"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            name="Ошибка Δ-ML CatBoost",
            x=formulas,
            y=ml_errs,
            marker_color="#4CAF50",
            showlegend=False,
            customdata=sources,
            hovertemplate="<b>%{x}</b><br>Ошибка ML: %{y:.2f} эВ<br>Источник: %{customdata}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=30, t=70, b=40)
    )
    fig.update_yaxes(title_text="Ширина зоны Eg (эВ)", gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig.update_yaxes(title_text="Абсолютная ошибка |ΔEg| (эВ)", gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
    fig.update_xaxes(title_text="Полупроводник", gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig.update_xaxes(title_text="Полупроводник", gridcolor="rgba(0,0,0,0.06)", row=1, col=2)

    return fig


def plot_fermi_spectrum_interactive(isotope: str) -> go.Figure:
    """Строит непрерывный релятивистский спектр Ферми P(E) бета-электронов."""
    if isotope not in ISOTOPE_DECAY_PARAMS:
        return go.Figure()
    
    pdf, norm, mean_e = compute_normalized_fermi_spectrum(isotope)
    e_max = ISOTOPE_DECAY_PARAMS[isotope]["e_max_kev"]
    
    e_arr = np.linspace(0.1, e_max, 250)
    p_arr = [pdf(e) for e in e_arr]
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=e_arr,
            y=p_arr,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#1f77b4", width=2.5),
            fillcolor="rgba(31, 119, 180, 0.2)",
            name=f"Спектр Ферми {isotope}",
            hovertemplate="Энергия E: %{x:.2f} кэВ<br>Плотность вер. P(E): %{y:.5f} кэВ⁻¹<extra></extra>"
        )
    )
    
    fig.add_vline(
        x=mean_e,
        line_dash="dash",
        line_color="#D9534F",
        line_width=2,
        annotation_text=f"&lt;E_beta&gt; = {mean_e:.2f} кэВ",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=e_max,
        line_dash="dot",
        line_color="#333333",
        line_width=1.5,
        annotation_text=f"E_max = {e_max:.1f} кэВ",
        annotation_position="top left"
    )
    
    fig.update_layout(
        template="plotly_white",
        title=f"Релятивистский спектр Ферми бета-электронов ({isotope})",
        xaxis=dict(title="Кинетическая энергия бета-электрона E (кэВ)", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="Нормированная вероятность P(E) (кэВ⁻¹)", gridcolor="rgba(0,0,0,0.06)"),
        height=360,
        margin=dict(l=50, r=40, t=60, b=40)
    )
    return fig


def plot_ode_stopping_profile(
    initial_energy_kev: float,
    density: float,
    z_eff: float,
    a_eff: float,
    material_name: str,
    isotope: str
) -> go.Figure:
    """Строит численное решение ОДУ замедления E(x) и профиль ионизационных потерь dE/dx(x)."""
    t_span, e_vals = solve_energy_loss_ode(
        initial_energy_kev, density, z_eff, a_eff, max_distance_um=50.0, num_points=150
    )
    
    sp_vals = [joy_luo_stopping_power_exact(e, density, z_eff, a_eff) for e in e_vals]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"<b>Кривая замедления E(x) в {material_name}</b>",
            f"<b>Профиль ионизационных потерь -dE/dx(x)</b>"
        ],
        horizontal_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(
            x=t_span,
            y=e_vals,
            mode="lines",
            line=dict(color="#0275D8", width=2.5),
            name="Энергия E(x)",
            hovertemplate="Глубина x: %{x:.3f} мкм<br>Энергия E: %{y:.2f} кэВ<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t_span,
            y=sp_vals,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#D9534F", width=2.5),
            fillcolor="rgba(217, 83, 79, 0.2)",
            name="Потери -dE/dx",
            hovertemplate="Глубина x: %{x:.3f} мкм<br>-dE/dx: %{y:.2f} кэВ/мкм<extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_white",
        height=360,
        showlegend=False,
        margin=dict(l=40, r=30, t=60, b=40)
    )
    fig.update_xaxes(title_text="Глубина проникновения x (мкм)", gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig.update_xaxes(title_text="Глубина проникновения x (мкм)", gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
    fig.update_yaxes(title_text="Остаточная энергия E (кэВ)", gridcolor="rgba(0,0,0,0.06)", row=1, col=1)
    fig.update_yaxes(title_text="Ионизационные потери -dE/dx (кэВ/мкм)", gridcolor="rgba(0,0,0,0.06)", row=1, col=2)
    
    return fig


def plot_displacement_cross_section_curve(
    z_target: float,
    a_target: float,
    ed_ev: float,
    isotope: str,
    material_name: str
) -> go.Figure:
    """Строит сечение дефектообразования Мотта / Мак-Кинли–Фешбаха sigma_d(E)."""
    e_max = ISOTOPE_DECAY_PARAMS.get(isotope, {}).get("e_max_kev", 66.9)
    
    m_nuc = a_target * M_U_C2_KEV
    c_term = ed_ev * m_nuc / 1000.0
    disc = (4.0 * M_E_C2_KEV)**2 + 8.0 * c_term
    e_thresh_kev = (-4.0 * M_E_C2_KEV + np.sqrt(disc)) / 4.0
    
    max_e_plot = max(e_max * 1.25, e_thresh_kev * 1.2 if e_thresh_kev < 500 else e_max * 1.25)
    e_arr = np.linspace(1.0, max_e_plot, 250)
    sigma_vals = [mckinley_feshbach_displacement_cross_section(e, z_target, a_target, ed_ev) for e in e_arr]
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=e_arr,
            y=sigma_vals,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#6f42c1", width=2.5),
            fillcolor="rgba(111, 66, 193, 0.15)",
            name=f"Сечение σ_d(E)",
            hovertemplate="Энергия E: %{x:.1f} кэВ<br>Сечение σ_d: %{y:.2f} барн<extra></extra>"
        )
    )
    
    fig.add_vline(
        x=e_thresh_kev,
        line_dash="dash",
        line_color="#E57373",
        line_width=2,
        annotation_text=f"Порог дефектов: {e_thresh_kev:.1f} кэВ",
        annotation_position="top left"
    )
    
    fig.add_vline(
        x=e_max,
        line_dash="dot",
        line_color="#2E7D32",
        line_width=2,
        annotation_text=f"E_max ({isotope}) = {e_max:.1f} кэВ",
        annotation_position="top right"
    )
    
    is_immune = (e_max < e_thresh_kev)
    status_text = "Радиационная неуязвимость (E_max &lt; E_thresh)" if is_immune else "Возможно дефектообразование (E_max &gt; E_thresh)"
    
    fig.update_layout(
        template="plotly_white",
        title=f"Сечение дефектообразования: {material_name} ({status_text})",
        xaxis=dict(title="Кинетическая энергия бета-электрона E (кэВ)", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="Сечение смещения атомов σ_d (барны)", gridcolor="rgba(0,0,0,0.06)"),
        height=360,
        margin=dict(l=50, r=40, t=60, b=40)
    )
    return fig