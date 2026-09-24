import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
pd.set_option("styler.render.max_elements", 1_000_000)
from src.screening.ranker import BetavoltaicLibrary
from src.screening.pareto import identify_pareto_frontier_3d
from app.plots import plot_pareto_interactive, plot_3d_materials_space, CRYSTAL_SYSTEMS_RU

# Настройка страницы (чистый академический вид без эмодзи)
st.set_page_config(
    page_title="Скрининг бетавольтаических полупроводников",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Словарь перевода технических названий колонок
COLUMN_MAPPING = {
    "formula": "Формула",
    "mp_id": "ID Materials Project",
    "crystal_system": "Сингония",
    "material_class": "Класс материала",
    "is_viable": "Химическая стойкость",
    "density": "Плотность (г/см³)",
    "volume": "Объем ячейки (Å³)",
    "e_above_hull": "Энергия над оболочкой (эВ)",
    "band_gap_dft": "Eg DFT (эВ)",
    "delta_eg_predicted": "Поправка ΔEg (эВ)",
    "band_gap_calibrated": "Eg калибр. (эВ)",
    "eps_ehp_ev": "Энергия пары (эВ)",
    "Voc_est_v": "Оценка Voc (В)",
    "theoretical_efficiency_pct": "Теор. КПД (%)",
    "ed_est_ev": "Порог смещения Ed (эВ)",
    "radiation_resistance_score": "Индекс стойкости (0–100)",
    "penetration_depth_um_Ni63": "Глубина пробега Ni-63 (мкм)",
    "carriers_per_electron_Ni63": "Пар на электрон Ni-63",
    "t_max_ev_Ni63": "T_max отдачи Ni-63 (эВ)",
    "is_immune_Ni63": "Иммунитет к Ni-63",
    "penetration_depth_um_H3": "Глубина пробега H-3 (мкм)",
    "carriers_per_electron_H3": "Пар на электрон H-3",
    "t_max_ev_H3": "T_max отдачи H-3 (эВ)",
    "is_immune_H3": "Иммунитет к H-3",
    "penetration_depth_um_C14": "Глубина пробега C-14 (мкм)",
    "carriers_per_electron_C14": "Пар на электрон C-14",
    "t_max_ev_C14": "T_max отдачи C-14 (эВ)",
    "is_immune_C14": "Иммунитет к C-14",
    "penetration_depth_um_Pm147": "Глубина пробега Pm-147 (мкм)",
    "carriers_per_electron_Pm147": "Пар на электрон Pm-147",
    "t_max_ev_Pm147": "T_max отдачи Pm-147 (эВ)",
    "is_immune_Pm147": "Иммунитет к Pm-147",
}

# Кэшируем загрузку данных из SQLite
@st.cache_data
def load_data():
    project_root = Path(__file__).resolve().parents[1]
    lib = BetavoltaicLibrary()
    df = lib.get_full_dataframe()
    pareto_csv = project_root / "reports" / "figures" / "pareto_champions.csv"
    pareto_df = pd.read_csv(pareto_csv) if pareto_csv.exists() else df.head(10)
    return df, pareto_df

df, pareto_df = load_data()

# Заголовок системы
st.title("Информационно-аналитическая система скрининга бетавольтаических материалов")
st.caption("Магистерская диссертация: «Разработка библиотеки бетавольтаических материалов» | Самарский Политех (СамГТУ)")

# Метрики в шапке
col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего материалов в базе", f"{len(df):,}")
col2.metric("Стойких полупроводников", f"{(df['is_viable'] == 1).sum():,}")
col3.metric("Точность ML-калибровки R²", "0.711", delta="+36.5% к DFT")
col4.metric("Доступных изотопов", "4 (Ni-63, H-3, C-14, Pm-147)")

st.divider()

# --- Боковая панель (Фильтры поиска) ---
st.sidebar.header("Параметры отбора")

selected_isotope = st.sidebar.selectbox(
    "Радиоактивный изотоп:",
    ["Ni-63", "H-3", "C-14", "Pm-147"],
    index=0,
    help="Радиоактивный источник для расчета пробега бета-частиц, энерговыделения и кинематики отдачи."
)

clean_tag = selected_isotope.replace("-", "")

# Фильтр химической жизнеспособности
only_viable = st.sidebar.checkbox(
    "Только химически стойкие полупроводники",
    value=True,
    help="Исключает растворимые соли (галогениды), гидриды, гидроксиды, токсичные цианиды, нестабильные ацетилиды и изолирующую диэлектрическую керамику"
)

# Фильтр по классам полупроводников
material_classes = [
    "Все классы",
    "Ковалентные (IV, III-V, карбиды, бориды, нитриды)",
    "Оксидные полупроводники (простые и тройные)",
    "Халькогениды (II-VI, дихалькогениды)",
    "Сложные оксиды",
    "Сложные халькогениды",
    "Прочие полупроводники"
]
selected_class = st.sidebar.selectbox(
    "Класс полупроводников:",
    material_classes,
    index=0
)

# Ползунки фильтрации
min_eff = st.sidebar.slider("Минимальный теоретический КПД (%)", 1.0, 22.0, 12.0, 0.5)
min_rad = st.sidebar.slider("Минимальный индекс радиационной стойкости", 0.0, 100.0, 20.0, 5.0)
gap_range = st.sidebar.slider("Диапазон запрещенной зоны Eg (эВ)", 0.5, 8.0, (1.1, 5.5), 0.1)

# Фильтр по максимальной глубине пробега с динамической калибровкой под изотоп
depth_col_curr = f"penetration_depth_um_{clean_tag}"
if depth_col_curr in df.columns:
    min_depth_data = float(df[depth_col_curr].min())
    max_depth_bound = float(df[depth_col_curr].quantile(0.99))
    if selected_isotope == "H-3":
        min_val_slider = 0.05
        step_slider = 0.01
        max_val_slider = max(0.2, round(max_depth_bound, 2))
    elif selected_isotope == "Ni-63":
        min_val_slider = 0.5
        step_slider = 0.1
        max_val_slider = round(max_depth_bound, 1)
    else:  # C-14, Pm-147
        min_val_slider = 1.0
        step_slider = 0.5
        max_val_slider = round(max_depth_bound, 1)
else:
    min_val_slider, max_val_slider, step_slider = 0.1, 10.0, 0.1

max_depth_slider = st.sidebar.slider(
    f"Макс. глубина пробега {selected_isotope} (мкм)",
    min_value=min_val_slider,
    max_value=max_val_slider,
    value=max_val_slider,
    step=step_slider,
    help="Максимально допустимая толщина полупроводника для полного поглощения энергии бета-частиц"
)

# Фильтр по формуле
search_formula = st.sidebar.text_input("Поиск по химической формуле (например: SiC, GaN, C, TiO2, BN):", "").strip()

# Фильтрация данных
filtered_df = df[
    (df["theoretical_efficiency_pct"] >= min_eff) &
    (df["radiation_resistance_score"] >= min_rad) &
    (df["band_gap_calibrated"] >= gap_range[0]) &
    (df["band_gap_calibrated"] <= gap_range[1]) &
    (df[depth_col_curr] <= max_depth_slider)
].copy()

if only_viable:
    filtered_df = filtered_df[filtered_df["is_viable"] == 1]

if selected_class != "Все классы":
    filtered_df = filtered_df[filtered_df["material_class"] == selected_class]

if search_formula:
    filtered_df = filtered_df[filtered_df["formula"].str.contains(search_formula, case=False, na=False)]

st.sidebar.info(f"Найдено материалов по критериям: **{len(filtered_df)}** из {len(df)}")

# Динамический расчет 3D Парето-чемпионов под текущую фильтрацию и выбранный изотоп
if not filtered_df.empty:
    depth_col = f"penetration_depth_um_{clean_tag}"
    pts = filtered_df[["theoretical_efficiency_pct", "radiation_resistance_score", depth_col]].values
    pareto_mask = identify_pareto_frontier_3d(pts)
    active_pareto_df = filtered_df[pareto_mask].sort_values(by="theoretical_efficiency_pct", ascending=False)
else:
    active_pareto_df = pd.DataFrame()

# --- Вкладки основного окна ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Парето-скрининг", 
    "База материалов", 
    "3D Пространство свойств", 
    "Физические и ML-модели"
])

# -------------------------------------------------------------
# ВКЛАДКА 1: ПАРЕТО-СКРИНИНГ
# -------------------------------------------------------------
with tab1:
    st.subheader(f"Карта многокритериального отбора (Изотоп: {selected_isotope})")
    
    if filtered_df.empty:
        st.warning("⚠️ По заданным критериям фильтрации материалы не найдены. Ослабьте фильтры на боковой панели.")
    
    st.plotly_chart(plot_pareto_interactive(filtered_df, active_pareto_df, selected_isotope), use_container_width=True)
    
    if not active_pareto_df.empty:
        st.caption(f"Выделено 3D Парето-чемпионов в активной выборке: **{len(active_pareto_df)}** материалов (помечены красными звёздами).")

    with st.expander("Физический смысл и математическая инвариантность 3D Парето-отбора", expanded=False):
        st.markdown(r"""
        * **Ось X (Теоретический КПД, %)** — предельная эффективность прямого бетавольтаического преобразования (модель Кляйна для генерации пар и предел Шокли-Квиссера–Олсена). Зависит от ширины зоны $E_g$.
        * **Ось Y (Радиационная стойкость R_score, %)** — термодинамическая стойкость решетки к дефектообразованию относительно эталонного алмаза ($E_d / E_d^{алмаз} \cdot 100\%$). Зависит от энергии когезии $E_{coh}$ и $E_g$.
        * **Глубина пробега (Пробег R, мкм)** — толщина слоя полного поглощения энергии бета-электронов по модели Фельдмана: $R = 0.04 \cdot E_\beta^{1.75} / \rho$.
        * **Почему набор чемпионов инвариантен к изотопу?**  
          Поскольку для фиксированного изотопа множитель $0.04 \cdot E_\beta^{1.75}$ является положительной константой, минимизация пробега $\min R$ строго эквивалентна максимизации плотности полупроводника $\max \rho$. Монотонное масштабирование не нарушает отношений Парето-доминирования, поэтому качественный список лидеров стабилен, в то время как численные эксплуатационные параметры (толщина слоя $R$ в мкм и генерация пар $N_{pairs}$) пересчитываются строго под выбранный изотоп.
        * **Красные звёздочки (3D Парето-чемпионы)** — недоминируемые материалы в пространстве $\max \eta_{max}, \max R_{score}, \min R$.
        """)

    st.markdown("### Топ-5 рекомендуемых материалов по многокритериальному рангу")
    if not filtered_df.empty:
        top_cols = [
            "formula", "mp_id", "material_class", "crystal_system", "density", 
            "band_gap_calibrated", "theoretical_efficiency_pct", 
            "radiation_resistance_score", f"penetration_depth_um_{clean_tag}", f"carriers_per_electron_{clean_tag}"
        ]
        filtered_df["_rank_score"] = filtered_df["theoretical_efficiency_pct"] * 0.5 + filtered_df["radiation_resistance_score"] * 0.5
        display_top = filtered_df.sort_values("_rank_score", ascending=False)[top_cols].head(5).copy()
        display_top["crystal_system"] = display_top["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(display_top["crystal_system"])
        display_top = display_top.rename(columns=COLUMN_MAPPING)
        display_top = display_top.round(2)
        st.dataframe(display_top, use_container_width=True, hide_index=True)
    else:
        st.info("Нет данных для формирования списка Топ-5.")

# -------------------------------------------------------------
# ВКЛАДКА 2: БАЗА МАТЕРИАЛОВ
# -------------------------------------------------------------
with tab2:
    st.subheader("Интерактивная таблица библиотеки полупроводников")
    
    if filtered_df.empty:
        st.warning("⚠️ Выборка пуста. Пожалуйста, измените параметры фильтрации в боковой панели.")
    else:
        table_cols = [c for c in [
            "formula", "mp_id", "material_class", "crystal_system", "density", 
            "band_gap_dft", "delta_eg_predicted", "band_gap_calibrated", 
            "theoretical_efficiency_pct", "ed_est_ev", "radiation_resistance_score",
            f"penetration_depth_um_{clean_tag}", f"carriers_per_electron_{clean_tag}"
        ] if c in filtered_df.columns]
        
        table_df = filtered_df[table_cols].copy()
        table_df["crystal_system"] = table_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(table_df["crystal_system"])
        table_df = table_df.rename(columns=COLUMN_MAPPING)
        table_df = table_df.round(2)
        st.dataframe(table_df, use_container_width=True, height=380, hide_index=True)
        
        csv_data = table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать выборку (CSV)",
            data=csv_data,
            file_name=f"betavoltaic_screening_{selected_isotope}.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Паспорт выбранного полупроводника")
    available_formulas = filtered_df["formula"].dropna().unique().tolist() if not filtered_df.empty else []
    if available_formulas:
        default_idx = available_formulas.index("BN") if "BN" in available_formulas else (available_formulas.index("SiC") if "SiC" in available_formulas else 0)
        chosen_mat = st.selectbox("Выберите соединение для детального анализа:", available_formulas, index=default_idx)
        mat_row = filtered_df[filtered_df["formula"] == chosen_mat].iloc[0]
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Теоретический КПД", f"{mat_row['theoretical_efficiency_pct']:.2f} %")
        c_m2.metric("Радиационная стойкость", f"{mat_row['radiation_resistance_score']:.1f} % (отн. алмаза)")
        c_m3.metric("Калиброванная зона Eg", f"{mat_row['band_gap_calibrated']:.2f} эВ", delta=f"DFT: {mat_row['band_gap_dft']:.2f} эВ")
        c_m4.metric(f"Глубина пробега ({selected_isotope})", f"{mat_row.get(f'penetration_depth_um_{clean_tag}', 0.0):.2f} мкм")
        
        t_max_val = mat_row.get(f't_max_ev_{clean_tag}', 0.0)
        ed_val = mat_row.get('ed_est_ev', 25.0)
        is_immune_val = (t_max_val < ed_val)
        immune_str = "Полная радиационная неуязвимость ($T_{max} < E_d$, упругое смещение узлов невозможно)" if is_immune_val else f"Возможно образование дефектов смещения ($T_{{max}} = {t_max_val:.1f} > E_d = {ed_val:.1f}$ эВ)"

        st.info(f"Материал **{chosen_mat}** ({CRYSTAL_SYSTEMS_RU.get(mat_row['crystal_system'], mat_row['crystal_system'])} сингония, класс: {mat_row.get('material_class', 'Полупроводник')}, плотность {mat_row['density']:.2f} г/см³). Порог образования радиационных дефектов $E_d \\approx {ed_val:.1f}$ эВ (модель Келли–Гроувса). Макс. энергия отдачи ядра для {selected_isotope}: $T_{{max}} \\approx {t_max_val:.1f}$ эВ. **Статус стойкости:** {immune_str}. При поглощении одного бета-электрона изотопа {selected_isotope} генерируется в среднем **{int(mat_row.get(f'carriers_per_electron_{clean_tag}', 0))}** электронно-дырочных пар.")
    else:
        st.info("Выберите другие параметры фильтрации для просмотра паспорта материала.")

# -------------------------------------------------------------
# ВКЛАДКА 3: 3D ПРОСТРАНСТВО СВОЙСТВ
# -------------------------------------------------------------
with tab3:
    st.subheader(f"3D Пространство критериев Парето-оптимизации ({selected_isotope})")
    if filtered_df.empty:
        st.warning("⚠️ Нет данных для построения 3D пространства свойств.")
    st.plotly_chart(plot_3d_materials_space(filtered_df, active_pareto_df, selected_isotope), use_container_width=True)
    st.caption("3D-пространство визуализирует фундаментальный компромисс Парето-отбора: максимизацию КПД (ось X), максимизацию радиационной стойкости (ось Y) и минимизацию необходимой толщины кристалла (ось Z).")

# -------------------------------------------------------------
# ВКЛАДКА 4: ФИЗИЧЕСКИЕ И ML-МОДЕЛИ
# -------------------------------------------------------------
with tab4:
    st.subheader("Физико-математические основы и интерпретация ML-калибровки")
    
    col_a, col_b = st.columns(2)
    project_root = Path(__file__).resolve().parents[1]
    shap_path = project_root / "reports" / "figures" / "shap_summary.png"
    shap_bar_path = project_root / "reports" / "figures" / "shap_importance_bar.png"
    stopping_path = project_root / "reports" / "figures" / "stopping_power_curves.png"
    
    with col_a:
        st.markdown("#### Анализ значимости дескрипторов (SHAP / XAI)")
        if shap_path.exists():
            st.image(str(shap_path), use_container_width=True, caption="Распределение SHAP-значений по дескрипторам Magpie")
        if shap_bar_path.exists():
            with st.expander("Показать столбчатую диаграмму средней важности (|SHAP|)", expanded=False):
                st.image(str(shap_bar_path), use_container_width=True, caption="Рейтинг топ-15 наиболее влиятельных признаков")
        st.markdown("""
        **Интерпретация результатов:**
        * График показывает вклад физико-химических дескрипторов кристалла в прогнозирование величины поправки $\\Delta E_g = E_g^{exp} - E_g^{DFT}$.
        * Красные точки соответствуют высоким значениям признака, синие — низким.
        * Наибольший вклад в коррекцию систематической ошибки DFT вносят электроотрицательность элементов, радиусы атомов и плотность упаковки решетки.
        """)
        
    with col_b:
        st.markdown("#### Ионизационные потери энергии (Модель Бете-Блоха / Джоя-Ло)")
        if stopping_path.exists():
            st.image(str(stopping_path), use_container_width=True, caption="Кривые тормозной способности -dE/dx для эталонных полупроводников")
        st.markdown("""
        **Интерпретация результатов:**
        * График отражает удельную тормозную способность материалов $-\\frac{dE}{dx}$ (кэВ/мкм) по модификации Джоя-Ло.
        * Пик кривой соответствует области максимальной плотности генерации свободных носителей заряда.
        * Средняя глубина пробега электронов рассчитывается по формуле Фельдмана $R = 0.04 \\cdot E_\\beta^{1.75} / \\rho$, определяя оптимальную толщину базовой области $p\\text{--}n$-перехода.
        """)