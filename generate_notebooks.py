import json
from pathlib import Path

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
    print(f"  ✓ Ноутбук сгенерирован: {filepath}")

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

def main():
    print("📓 Автоматическое наполнение исследовательских Jupyter-ноутбуков...")
    nb_dir = Path("notebooks")
    nb_dir.mkdir(exist_ok=True)

    # 1. Ноутбук 01: Выгрузка и очистка
    cells_01 = [
        md_cell("# 01. Сбор и предобработка данных (ETL)\nДанный ноутбук иллюстрирует этапы сбора данных из Materials Project и очистки."),
        code_cell("import pandas as pd\n\nraw_df = pd.read_parquet('../data/01_raw/materials_raw.parquet')\nclean_df = pd.read_parquet('../data/02_intermediate/materials_cleaned.parquet')\n\nprint(f'Исходный объем выгрузки: {len(raw_df)} кристаллов')\nprint(f'После очистки от ядов и дубликатов: {len(clean_df)} кристаллов')"),
        code_cell("clean_df[['formula', 'crystal_system', 'density', 'band_gap_dft', 'e_above_hull']].head(10)")
    ]
    create_notebook(cells_01, nb_dir / "01_data_extraction.ipynb")

    # 2. Ноутбук 02: EDA и кластеризация (Разделы 8.5.1 - 8.5.3 плана)
    cells_02 = [
        md_cell("# 02. Разведочный анализ данных (EDA) и кластеризация\nАнализ распределений свойств, корреляций и кластеров кристаллов."),
        code_cell("import sqlite3\nimport pandas as pd\nimport plotly.express as px\n\nconn = sqlite3.connect('../data/05_database/betavoltaic_library.db')\ndf = pd.read_sql('SELECT m.formula, m.crystal_system, m.density, e.band_gap_calibrated, e.theoretical_efficiency_pct, p.ed_est_ev FROM materials m JOIN electronic_properties e ON m.mp_id = e.mp_id JOIN betavoltaic_performance p ON m.mp_id = p.mp_id', conn)\nconn.close()\n\ndf.head()"),
        code_cell("# Распределение материалов по сингониям кристаллической решетки\nfig_pie = px.pie(df, names='crystal_system', title='Распределение полупроводников по сингониям')\nfig_pie.show()"),
        code_cell("# Корреляция запрещенной зоны и плотности\nfig_scatter = px.scatter(df.sample(2000), x='band_gap_calibrated', y='theoretical_efficiency_pct', color='crystal_system', title='Корреляция Eg и теоретического КПД')\nfig_scatter.show()")
    ]
    create_notebook(cells_02, nb_dir / "02_eda_and_clustering.ipynb")

    # 3. Ноутбук 03: ML-эксперименты и SHAP (Разделы 8.3.8, 9.7)
    cells_03 = [
        md_cell("# 03. Результаты машинного обучения и объяснимость\nДемонстрация работы модели CatBoost, метрик валидации и значений SHAP."),
        code_cell("import json\n\nwith open('../models/metrics_report.json', 'r') as f:\n    metrics = json.load(f)\n\nprint('Итоговые метрики модели калибровки:')\nfor k, v in metrics.items():\n    print(f'  • {k}: {v}')"),
        code_cell("from IPython.display import Image\nImage(filename='../reports/figures/shap_summary.png')")
    ]
    create_notebook(cells_03, nb_dir / "03_ml_experiments.ipynb")

    # 4. Ноутбук 04: Парето-скрининг (Раздел 8.5.4)
    cells_04 = [
        md_cell("# 04. Многокритериальный Парето-скрининг\nФинальный отбор полупроводников-чемпионов под различные изотопы."),
        code_cell("import pandas as pd\n\nchampions = pd.read_csv('../reports/figures/pareto_champions.csv')\nprint(f'Выявлено Парето-чемпионов: {len(champions)}')\nchampions"),
        code_cell("from IPython.display import Image\nImage(filename='../reports/figures/stopping_power_curves.png')")
    ]
    create_notebook(cells_04, nb_dir / "04_pareto_screening.ipynb")

    print("\n Все 4 ноутбука готовы к запуску и демонстрации!")

if __name__ == "__main__":
    main()