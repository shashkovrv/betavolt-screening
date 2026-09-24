import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Защита путей для кнопки Play
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import matplotlib
matplotlib.use("Agg")  # Рендеринг без GUI-окон для надежного сохранения
import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostRegressor

# Словарь перевода базовых дескрипторов
BASE_TRANSLATION = {
    "band_gap_dft": "Ширина зоны DFT (эВ)",
    "density": "Плотность структуры (г/см³)",
    "volume": "Объем ячейки (Å³)",
}

STAT_TRANSLATION = {
    "mean": "Ср.",
    "avg_dev": "Ср. откл.",
    "range": "Диапазон",
    "maximum": "Макс.",
    "minimum": "Мин.",
    "mode": "Мода",
}

PROP_TRANSLATION = {
    "Number": "атомного номера",
    "MendeleevNumber": "числа Менделеева",
    "AtomicWeight": "атомной массы",
    "MeltingT": "темп. плавления",
    "Column": "номера группы",
    "Row": "номера периода",
    "CovalentRadius": "ковалентного радиуса",
    "Electronegativity": "электроотрицательности",
    "NsValence": "числа s-электронов",
    "NpValence": "числа p-электронов",
    "NdValence": "числа d-электронов",
    "NfValence": "числа f-электронов",
    "NValence": "валентных электронов",
    "NsUnfilled": "незаполненных s-орбиталей",
    "NpUnfilled": "незаполненных p-орбиталей",
    "NdUnfilled": "незаполненных d-орбиталей",
    "NfUnfilled": "незаполненных f-орбиталей",
    "NUnfilled": "незаполненных орбиталей",
    "GSvolume_pa": "объема на атом",
    "GSbandgap": "Eg основного состояния",
    "GSmagmom": "магнитного момента",
    "SpaceGroupNumber": "номера простр. группы",
}


def translate_feature_name(col: str) -> str:
    if col in BASE_TRANSLATION:
        return BASE_TRANSLATION[col]
    if col.startswith("MagpieData "):
        parts = col.replace("MagpieData ", "").split(" ", 1)
        if len(parts) == 2:
            stat, prop = parts[0], parts[1]
            stat_ru = STAT_TRANSLATION.get(stat, stat)
            prop_ru = PROP_TRANSLATION.get(prop, prop)
            return f"{stat_ru} {prop_ru}"
    return col


def run_explainability_analysis(
    features_path: str = "data/03_features/materials_features.parquet",
    experimental_path: str = "data/04_external/experimental_bandgaps.parquet",
    model_path: str = "models/delta_eg_catboost.cbm",
    output_summary_path: str = "reports/figures/shap_summary.png",
    output_bar_path: str = "reports/figures/shap_importance_bar.png"
):
    print("[*] Запуск анализа объяснимости модели (XAI / SHAP для Delta-Learning)...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели {model_path} не найден! Сначала запусти delta_learner.py.")

    # 1. Загрузка данных и модели
    df_features = pd.read_parquet(features_path)
    df_exp = pd.read_parquet(experimental_path)
    
    model = CatBoostRegressor()
    model.load_model(model_path)

    # 2. Формируем ту же матрицу признаков, на которой обучалась модель
    magpie_cols = [c for c in df_features.columns if "MagpieData" in c]
    all_feature_cols = ["band_gap_dft", "density", "volume"] + magpie_cols

    train_df = pd.merge(
        df_features,
        df_exp[["clean_formula", "band_gap_exp"]],
        left_on="formula",
        right_on="clean_formula",
        how="inner"
    ).drop_duplicates(subset=["clean_formula"]).reset_index(drop=True)

    X = train_df[all_feature_cols]

    # 3. Инициализация SHAP TreeExplainer для CatBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Переводим имена колонок в датафрейме для аккуратных графиков
    ru_cols = [translate_feature_name(c) for c in all_feature_cols]
    X_ru = X.copy()
    X_ru.columns = ru_cols

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)

    # 4. График 1: SHAP Beeswarm Plot (Влияние величины признака на прогноз поправки Delta_Eg)
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X_ru, show=False, max_display=12)
    plt.title("Влияние физико-химических дескрипторов на квантовую поправку $\\Delta E_g$ (SHAP)", fontsize=13, pad=15)
    plt.xlabel("Влияние на значение поправки $\\Delta E_g = E_g^{exp} - E_g^{DFT}$ (эВ)", fontsize=11)
    
    # Русификация шкалы цветовой легенды (Colorbar)
    fig = plt.gcf()
    for ax in fig.axes:
        if ax.get_ylabel() == "Feature value":
            ax.set_ylabel("Величина дескриптора", fontsize=11)
            ax.set_yticklabels(["Низкая", "Высокая"])

    plt.tight_layout()
    plt.savefig(output_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 5. График 2: Bar Plot абсолютной важности
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_ru, plot_type="bar", show=False, max_display=12)
    plt.title("Топ-12 ключевых дескрипторов квантовой недооценки $\\Delta E_g$", fontsize=13, pad=15)
    plt.xlabel("Среднее абсолютное влияние на модель $|\\mathrm{SHAP}|$ (эВ)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[+] Анализ объяснимости успешно завершен! Графики сохранены в reports/figures/")


if __name__ == "__main__":
    run_explainability_analysis()