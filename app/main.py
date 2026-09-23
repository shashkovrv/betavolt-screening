import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from src.screening.ranker import BetavoltaicLibrary
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
    "penetration_depth_um_H3": "Глубина пробега H-3 (мкм)",
    "carriers_per_electron_H3": "Пар на электрон H-3",
    "penetration_depth_um_C14": "Глубина пробега C-14 (мкм)",
    "carriers_per_electron_C14": "Пар на электрон C-14",
    "penetration_depth_um_Pm147": "Глубина пробега Pm-147 (мкм)",
    "carriers_per_electron_Pm147": "Пар на электрон Pm-147",
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
col2.metric("Парето-чемпионов", f"{len(pareto_df)}")
col3.metric("Точность ML-калибровки R²", "0.691", delta="+35.7% к DFT")
col4.metric("Доступных изотопов", "4 (Ni-63, H-3, C-14, Pm-147)")

st.divider()

# --- Боковая панель (Фильтры поиска) ---
st.sidebar.header("Параметры отбора")

selected_isotope = st.sidebar.selectbox(
    "Радиоактивный изотоп:",
    ["Ni-63", "H-3", "C-14", "Pm-147"],
    index=0,
    help="Радиоактивный источник для расчета пробега бета-частиц и ионизационных потерь."
)

clean_tag = selected_isotope.replace("-", "")

# Ползунки фильтрации
min_eff = st.sidebar.slider("Минимальный теоретический КПД (%)", 5.0, 25.0, 18.0, 0.5)
min_rad = st.sidebar.slider("Минимальный индекс радиационной стойкости", 0.0, 100.0, 20.0, 5.0)
gap_range = st.sidebar.slider("Диапазон запрещенной зоны Eg (эВ)", 0.5, 10.0, (1.2, 5.5), 0.1)

# Фильтр по формуле
search_formula = st.sidebar.text_input("Поиск по химической формуле (например: SiC, GaN, C, TiO2):", "").strip()

# Фильтрация данных
filtered_df = df[
    (df["theoretical_efficiency_pct"] >= min_eff) &
    (df["radiation_resistance_score"] >= min_rad) &
    (df["band_gap_calibrated"] >= gap_range[0]) &
    (df["band_gap_calibrated"] <= gap_range[1])
].copy()

if search_formula:
    filtered_df = filtered_df[filtered_df["formula"].str.contains(search_formula, case=False, na=False)]

st.sidebar.info(f"Найдено материалов по критериям: **{len(filtered_df)}** из {len(df)}")

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
    
    st.plotly_chart(plot_pareto_interactive(filtered_df, pareto_df, selected_isotope), use_container_width=True)
    
    with st.expander("Физический смысл и правила чтения графика", expanded=False):
        st.markdown("""
        * **Ось X (Теоретический КПД, %)** — предельная эффективность преобразования кинетической энергии бета-электронов в электрический ток (комбинация модели Кляйна для генерации электронно-дырочных пар и предела Шокли-Квиссера).
        * **Ось Y (Индекс радиационной стойкости, 0–100)** — устойчивость полупроводника к деградации решетки под радиационной бомбардировкой (рассчитана на основе порога смещения атомов $E_d$).
        * **Цвет точек** — ширина запрещенной зоны $E_g$ после ML-калибровки (устранение систематической погрешности DFT).
        * **Красные звёздочки (Парето-чемпионы)** — материалы Парето-фронта, у которых невозможно увеличить КПД без сопутствующей потери радиационной стойкости. Для скрытия или отображения чемпионов нажмите на пункт в легенде графика вверху справа.
        """)

    st.markdown("### Топ-5 рекомендуемых материалов по КПД")
    top_cols = [
        "formula", "mp_id", "crystal_system", "density", 
        "band_gap_calibrated", "theoretical_efficiency_pct", 
        "radiation_resistance_score", f"penetration_depth_um_{clean_tag}", f"carriers_per_electron_{clean_tag}"
    ]
    display_top = filtered_df[top_cols].sort_values("theoretical_efficiency_pct", ascending=False).head(5).copy()
    display_top["crystal_system"] = display_top["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(display_top["crystal_system"])
    display_top = display_top.rename(columns=COLUMN_MAPPING)
    st.dataframe(display_top.style.format(precision=2), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# ВКЛАДКА 2: БАЗА МАТЕРИАЛОВ
# -------------------------------------------------------------
with tab2:
    st.subheader("Интерактивная таблица библиотеки полупроводников")
    
    table_df = filtered_df.copy()
    table_df["crystal_system"] = table_df["crystal_system"].map(CRYSTAL_SYSTEMS_RU).fillna(table_df["crystal_system"])
    table_df = table_df.rename(columns=COLUMN_MAPPING)
    st.dataframe(table_df.style.format(precision=2), use_container_width=True, height=380, hide_index=True)
    
    csv_data = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать выборку (CSV)",
        data=csv_data,
        file_name=f"betavoltaic_screening_{selected_isotope}.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Паспорт выбранного полупроводника")
    available_formulas = filtered_df["formula"].dropna().unique().tolist()
    if available_formulas:
        default_idx = available_formulas.index("C") if "C" in available_formulas else 0
        chosen_mat = st.selectbox("Выберите соединение для детального анализа:", available_formulas, index=default_idx)
        mat_row = filtered_df[filtered_df["formula"] == chosen_mat].iloc[0]
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Теоретический КПД", f"{mat_row['theoretical_efficiency_pct']:.2f} %")
        c_m2.metric("Радиационная стойкость", f"{mat_row['radiation_resistance_score']:.1f} / 100")
        c_m3.metric("Калиброванная зона Eg", f"{mat_row['band_gap_calibrated']:.2f} эВ", delta=f"DFT: {mat_row['band_gap_dft']:.2f} эВ")
        c_m4.metric(f"Глубина пробега ({selected_isotope})", f"{mat_row.get(f'penetration_depth_um_{clean_tag}', 0.0):.2f} мкм")
        
        st.info(f"Материал **{chosen_mat}** ({CRYSTAL_SYSTEMS_RU.get(mat_row['crystal_system'], mat_row['crystal_system'])} сингония, плотность {mat_row['density']:.2f} г/см³). Порог образования радиационных дефектов $E_d \\approx {mat_row['ed_est_ev']:.1f}$ эВ. При поглощении одного бета-электрона изотопа {selected_isotope} генерируется в среднем **{int(mat_row.get(f'carriers_per_electron_{clean_tag}', 0))}** электронно-дырочных пар.")

# -------------------------------------------------------------
# ВКЛАДКА 3: 3D ПРОСТРАНСТВО СВОЙСТВ
# -------------------------------------------------------------
with tab3:
    st.subheader("3D-визуализация пространства полупроводниковых свойств")
    st.plotly_chart(plot_3d_materials_space(filtered_df), use_container_width=True)
    st.caption("Оси отражают многомерный компромисс параметров: ширину запрещенной зоны (Eg), теоретический КПД и радиационную стойкость.")

# -------------------------------------------------------------
# ВКЛАДКА 4: ФИЗИЧЕСКИЕ И ML-МОДЕЛИ
# -------------------------------------------------------------
with tab4:
    st.subheader("Физико-математические основы и интерпретация ML-калибровки")
    
    col_a, col_b = st.columns(2)
    project_root = Path(__file__).resolve().parents[1]
    shap_path = project_root / "reports" / "figures" / "shap_summary.png"
    stopping_path = project_root / "reports" / "figures" / "stopping_power_curves.png"
    
    with col_a:
        st.markdown("#### Анализ значимости дескрипторов (SHAP / XAI)")
        if shap_path.exists():
            st.image(str(shap_path), use_container_width=True)
        st.markdown("""
        **Интерпретация результатов:**
        * График показывает вклад физико-химических дескрипторов кристалла в прогнозирование величины поправки $\\Delta E_g = E_g^{exp} - E_g^{DFT}$.
        * Красные точки соответствуют высоким значениям признака, синие — низким.
        * Наибольший вклад в коррекцию систематической ошибки DFT вносят электроотрицательность элементов, радиусы атомов и плотность упаковки решетки.
        """)
        
    with col_b:
        st.markdown("#### Ионизационные потери энергии (Модель Бете-Блоха / Джоя-Ло)")
        if stopping_path.exists():
            st.image(str(stopping_path), use_container_width=True)
        st.markdown("""
        **Интерпретация результатов:**
        * График отражает удельную тормозную способность материалов $-\\frac{dE}{dx}$ (кэВ/мкм) в зависимости от энергии налетающего электрона.
        * Пик кривой соответствует области максимальной плотности генерации свободных носителей заряда.
        * Расчет позволяет определить необходимую толщину полупроводникового слоя для полного поглощения энергии изотопа без паразитного излучения.
        """)