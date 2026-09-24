import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"  [+] Ноутбук сгенерирован: {filepath}")

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

def main():
    print("[*] Генерация подробных исследовательских Jupyter-ноутбуков для магистерской диссертации...")
    nb_dir = Path("notebooks")
    nb_dir.mkdir(exist_ok=True)

    # 1. Ноутбук 01: Выгрузка и очистка
    cells_01 = [
        md_cell("""# 01. Сбор и предобработка данных (ETL Pipeline)
**Магистерская диссертация:** «Разработка библиотеки бетавольтаических материалов» (СамГТУ)

В данном ноутбуке демонстрируется процесс сбора кристаллохимических данных из Materials Project API, санитарной фильтрации, отсечения токсичных и радиоактивных элементов, а также формирования промежуточных Parquet-датасетов."""),
        code_cell("""import pandas as pd
import numpy as np

# Загрузка сырых и очищенных данных
raw_path = '../data/01_raw/materials_raw.parquet'
clean_path = '../data/02_intermediate/materials_cleaned.parquet'

raw_df = pd.read_parquet(raw_path)
clean_df = pd.read_parquet(clean_path)

print(f'Исходный объем выгрузки Materials Project: {len(raw_df):,} кристаллов')
print(f'После очистки от токсичных/нестабильных фаз: {len(clean_df):,} кристаллов')
print(f'Отфильтровано нежелательных структур: {len(raw_df) - len(clean_df):,}')"""),
        md_cell("### Статистика физических диапазонов до и после очистки"),
        code_cell("""summary_stats = clean_df[['density', 'volume', 'band_gap_dft', 'e_above_hull']].describe().round(3)
summary_stats"""),
        md_cell("### Распределение материалов по сингониям кристаллической решетки"),
        code_cell("""print('Распределение по типам сингоний:')
clean_df['crystal_system'].value_counts()""")
    ]
    create_notebook(cells_01, nb_dir / "01_data_extraction.ipynb")

    # 2. Ноутбук 02: EDA и кластеризация
    cells_02 = [
        md_cell("""# 02. Разведочный анализ данных (EDA) и исследование физико-химического пространства
**Магистерская диссертация:** «Разработка библиотеки бетавольтаических материалов» (СамГТУ)

Исследование корреляций между шириной запрещенной зоны $E_g$, плотностью $\\rho$, порогом радиационных повреждений $E_d$ и термодинамической устойчивостью $E_{hull}$."""),
        code_cell("""import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px

# Подключение к библиотеке SQLite
conn = sqlite3.connect('../data/05_database/betavoltaic_library.db')
df = pd.read_sql('''
    SELECT m.formula, m.crystal_system, m.material_class, m.density, m.is_viable,
           e.band_gap_dft, e.delta_eg_predicted, e.band_gap_calibrated, e.theoretical_efficiency_pct,
           p.ed_est_ev, p.radiation_resistance_score,
           p.penetration_depth_um_Ni63, p.carriers_per_electron_Ni63, p.t_max_ev_Ni63, p.is_immune_Ni63
    FROM materials m
    JOIN electronic_properties e ON m.mp_id = e.mp_id
    JOIN betavoltaic_performance p ON m.mp_id = p.mp_id
''', conn)
conn.close()

print(f'Всего загружено записей из библиотеки: {len(df):,}')
print(f'Жизнеспособных полупроводников (is_viable=1): {df["is_viable"].sum():,}')
df.head(10)"""),
        md_cell("### Распределение жизнеспособных полупроводников по классам"),
        code_cell("""class_counts = df[df['is_viable'] == 1]['material_class'].value_counts()
class_counts"""),
        md_cell("### Корреляция калиброванной запрещенной зоны и радиационной стойкости"),
        code_cell("""fig = px.scatter(
    df[df['is_viable'] == 1].sample(min(2000, len(df))),
    x='band_gap_calibrated',
    y='radiation_resistance_score',
    color='material_class',
    hover_name='formula',
    labels={'band_gap_calibrated': 'Eg калибр. (эВ)', 'radiation_resistance_score': 'Индекс стойкости R_score (%)'},
    title='Корреляция ширины зоны Eg и радиационной стойкости полупроводников'
)
fig.show()""")
    ]
    create_notebook(cells_02, nb_dir / "02_eda_and_clustering.ipynb")

    # 3. Ноутбук 03: ML-эксперименты, Канонический Delta-Learning и SHAP
    cells_03 = [
        md_cell("""# 03. Машинное обучение: Канонический Delta-Learning и объяснимость (XAI / SHAP)
**Магистерская диссертация:** «Разработка библиотеки бетавольтаических материалов» (СамГТУ)

В данном ноутбуке анализируется калибровка систематической недооценки ширины запрещенной зоны DFT (PBE) с помощью градиентного бустинга CatBoost на дескрипторах Magpie:
$$\\Delta E_g = E_g^{exp} - E_g^{DFT}$$
$$E_g^{calibrated} = E_g^{DFT} + \\Delta E_g^{predicted}$$"""),
        code_cell("""import json
import pandas as pd
from IPython.display import Image, display

with open('../models/metrics_report.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

print('=' * 65)
print('  МЕТРИКИ ВАЛИДАЦИИ КАНОНИЧЕСКОГО DELTA-LEARNING:')
print('=' * 65)
for k, v in metrics.items():
    print(f'  • {k:<30} : {v}')
print('=' * 65)"""),
        md_cell("### Анализ Шепли-значений (SHAP Beeswarm Plot)"),
        code_cell("""print('Визуализация влияния физико-химических дескрипторов на поправку Delta_Eg:')
display(Image(filename='../reports/figures/shap_summary.png'))"""),
        md_cell("### Рейтинг абсолютной важности дескрипторов (SHAP Bar Plot)"),
        code_cell("""print('Топ-12 ключевых дескрипторов ошибки квантовых расчетов DFT:')
display(Image(filename='../reports/figures/shap_importance_bar.png'))""")
    ]
    create_notebook(cells_03, nb_dir / "03_ml_experiments.ipynb")

    # 4. Ноутбук 04: Парето-скрининг
    cells_04 = [
        md_cell("""# 04. Многокритериальный 3D Парето-скрининг бетавольтаических материалов
**Магистерская диссертация:** «Разработка библиотеки бетавольтаических материалов» (СамГТУ)

Оптимизация в трехмерном пространстве критериев:
$$\\max \\eta_{theor} \\quad (\\text{КПД}), \\qquad \\max R_{score} \\quad (\\text{Радиационная стойкость}), \\qquad \\min R_\\beta \\quad (\\text{Глубина пробега})$$
а также учет релятивистского кинематического порога образования дефектов Френкеля $T_{max} < E_d$."""),
        code_cell("""import pandas as pd
from IPython.display import Image, display

champions = pd.read_csv('../reports/figures/pareto_champions.csv')
print(f'Всего выделено 3D Парето-чемпионов: {len(champions)}')
champions[['formula', 'mp_id', 'material_class', 'density', 'band_gap_calibrated', 'theoretical_efficiency_pct', 'ed_est_ev', 'radiation_resistance_score', 'penetration_depth_um_Ni63', 'is_immune_Ni63']].head(15)"""),
        md_cell("### Ионизационные потери энергии и тормозная способность (Joy-Luo / Bethe-Bloch)"),
        code_cell("""display(Image(filename='../reports/figures/stopping_power_curves.png'))"""),
        md_cell("### Сравнение поведения ключевых эталонов под 4 бета-изотопа"),
        code_cell("""benchmarks = champions[champions['formula'].isin(['C', 'Si', 'SiC', 'GaN', 'TiO2', 'BN', 'AlN', 'BP', 'B4C'])]
benchmarks[['formula', 'band_gap_calibrated', 'theoretical_efficiency_pct', 'radiation_resistance_score', 'penetration_depth_um_Ni63', 'penetration_depth_um_H3', 'penetration_depth_um_C14', 'penetration_depth_um_Pm147']]""")
    ]
    create_notebook(cells_04, nb_dir / "04_pareto_screening.ipynb")

    print("\n[+] Все 4 исследовательских ноутбука успешно пересозданы с академическим наполнением!")

if __name__ == "__main__":
    main()