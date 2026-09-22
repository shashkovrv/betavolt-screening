import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
from src.screening.ranker import BetavoltaicLibrary
from app.plots import plot_pareto_interactive, plot_3d_materials_space

# Настройка страницы
st.set_page_config(
    page_title="Бетавольтаика | Materials Library",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кэшируем загрузку данных из SQLite, чтобы дашборд летал моментально
@st.cache_data
def load_data():
    lib = BetavoltaicLibrary()
    df = lib.get_full_dataframe()
    pareto_df = pd.read_csv("reports/figures/pareto_champions.csv") if Path("reports/figures/pareto_champions.csv").exists() else df.head(10)
    return df, pareto_df

df, pareto_df = load_data()

# Заголовок
st.title(" Информационно-аналитическая система скрининга бетавольтаических материалов")
st.caption("Магистерская ВКР: Разработка библиотеки данных и методов ML для скрининга бетавольтаиков")

# Метрики в шапке
col1, col2, col3, col4 = st.columns(4)
col1.metric("Всего материалов в БД", f"{len(df):,}")
col2.metric("Парето-чемпионов", f"{len(pareto_df)}")
col3.metric("ML точность калибровки R²", "0.691", delta="+35.7% к DFT")
col4.metric("Доступных изотопов", "4 (Ni, H, C, Pm)")

st.divider()

# --- Боковая панель (Фильтры поиска) ---
st.sidebar.header(" Параметры скрининга")

selected_isotope = st.sidebar.selectbox(
    "Выберите радиоактивный изотоп:",
    ["Ni-63", "H-3", "C-14", "Pm-147"],
    index=0
)

# Ползунки фильтрации
min_eff = st.sidebar.slider("Минимальный теоретический КПД (%)", 10.0, 22.0, 18.0, 0.5)
min_rad = st.sidebar.slider("Мин. индекс радиационной стойкости", 0.0, 100.0, 20.0, 5.0)
gap_range = st.sidebar.slider("Диапазон запрещенной зоны Eg (эВ)", 1.0, 10.0, (1.2, 5.5))

# Фильтр по формуле
search_formula = st.sidebar.text_input("Быстрый поиск по формуле (например, SiC, GaN):", "").strip()

# Фильтрация данных
filtered_df = df[
    (df["theoretical_efficiency_pct"] >= min_eff) &
    (df["radiation_resistance_score"] >= min_rad) &
    (df["band_gap_calibrated"] >= gap_range[0]) &
    (df["band_gap_calibrated"] <= gap_range[1])
]

if search_formula:
    filtered_df = filtered_df[filtered_df["formula"].str.contains(search_formula, case=False, na=False)]

st.sidebar.info(f"Найдено подходящих кандидатов: **{len(filtered_df)}** из {len(df)}")

# --- Вкладки основного окна ---
tab1, tab2, tab3, tab4 = st.tabs([
    " Парето-скрининг", 
    " База материалов (Таблица)", 
    " 3D Пространство свойств", 
    " Физика и ML-калибровка"
])

with tab1:
    st.subheader(f"Карта многокритериального отбора (Изотоп: {selected_isotope})")
    st.plotly_chart(plot_pareto_interactive(filtered_df, pareto_df, selected_isotope), use_container_width=True)
    
    st.markdown("### 🏆 Топ-5 рекомендуемых кандидатов под выбранные параметры:")
    clean_tag = selected_isotope.replace("-", "")
    top_cols = [
        "formula", "mp_id", "crystal_system", "density", 
        "band_gap_calibrated", "theoretical_efficiency_pct", 
        "radiation_resistance_score", f"penetration_depth_um_{clean_tag}", f"carriers_per_electron_{clean_tag}"
    ]
    st.dataframe(filtered_df[top_cols].sort_values("theoretical_efficiency_pct", ascending=False).head(5), use_container_width=True)

with tab2:
    st.subheader("Интерактивная таблица библиотеки полупроводников")
    st.dataframe(filtered_df, use_container_width=True, height=450)
    
    # Кнопка скачивания выборки в CSV
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Скачать отфильтрованные данные (CSV)",
        data=csv_data,
        file_name=f"betavoltaic_screening_{selected_isotope}.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("3D-визуализация пространства полупроводниковых свойств")
    st.plotly_chart(plot_3d_materials_space(filtered_df), use_container_width=True)

with tab4:
    st.subheader("Результаты машинного обучения и физического моделирования")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("reports/figures/shap_summary.png", caption="SHAP-анализ важности физико-химических дескрипторов (Глава 2)")
    with col_b:
        st.image("reports/figures/stopping_power_curves.png", caption="Кривые торможения бета-электронов Бете-Блоха (Глава 1/2)")