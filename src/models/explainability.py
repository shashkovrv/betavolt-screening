import sys
from pathlib import Path

# Защита путей для кнопки Play
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import matplotlib
matplotlib.use("Agg")  # Рендеринг без GUI-окон для надежного сохранения
import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostRegressor


def run_explainability_analysis(
    features_path: str = "data/03_features/materials_features.parquet",
    experimental_path: str = "data/04_external/experimental_bandgaps.parquet",
    model_path: str = "models/delta_eg_catboost.cbm",
    output_summary_path: str = "reports/figures/shap_summary.png",
    output_bar_path: str = "reports/figures/shap_importance_bar.png"
):
    print(" Запуск анализа объяснимости модели (XAI / SHAP Analysis)...")

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
    print(f"  Анализируем влияние признаков на выборке из {len(X)} экспериментальных материалов...")

    # 3. Инициализация SHAP TreeExplainer для CatBoost
    print("  Вычисление значений Шепли (SHAP values)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)

    # 4. График 1: SHAP Beeswarm Plot (Влияние величины признака на прогноз)
    print("  [1/2] Построение Beeswarm-графика распределения SHAP...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X, show=False, max_display=12)
    plt.title("SHAP Beeswarm: Влияние дескрипторов на калибровку запрещенной зоны", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(output_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ График сохранен в: {output_summary_path}")

    # 5. График 2: Bar Plot абсолютной важности
    print("  [2/2] Построение графика глобальной важности признаков...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=12)
    plt.title("Топ-12 наиболее значимых дескрипторов (Mean |SHAP value|)", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(output_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ График сохранен в: {output_bar_path}")

    # 6. Аналитический вывод в консоль
    mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=all_feature_cols).sort_values(ascending=False)
    
    print("\n" + "=" * 65)
    print("  ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ ДЛЯ ГЛАВЫ 2 ДИПЛОМА:")
    print("=" * 65)
    print("Топ-5 ключевых факторов, определяющих величину поправки:")
    for i, (feat, val) in enumerate(mean_abs_shap.head(5).items(), 1):
        print(f"  {i}. {feat:<35} | Среднее влияние: {val:.4f} эВ")
    print("=" * 65 + "\n")
    print(" Анализ объяснимости успешно завершен!")


if __name__ == "__main__":
    run_explainability_analysis()