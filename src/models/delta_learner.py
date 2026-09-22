import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_apply_delta_model(
    features_path: str = "data/03_features/materials_features.parquet",
    experimental_path: str = "data/04_external/experimental_bandgaps.parquet",
    model_output_path: str = "models/delta_eg_catboost.cbm",
    output_calibrated_path: str = "data/03_features/materials_calibrated.parquet"
):
    print(" Запуск модуля машинного обучения (Physics-Informed ML)...")

    # 1. Загрузка данных
    df_features = pd.read_parquet(features_path)
    df_exp = pd.read_parquet(experimental_path)

    # 2. Формируем признаки: Magpie + ФУНДАМЕНТАЛЬНАЯ ФИЗИКА (DFT gap, density, volume)
    magpie_cols = [c for c in df_features.columns if "MagpieData" in c]
    physical_cols = ["band_gap_dft", "density", "volume"]
    all_feature_cols = physical_cols + magpie_cols

    print(f"  Признаков: {len(all_feature_cols)} (включая DFT параметры и 132 дескриптора Magpie)")

    # 3. Сопоставляем эксперименты с признаками БЕЗ дублирования колонок
    # Из df_exp берем ТОЛЬКО clean_formula и band_gap_exp
    train_df = pd.merge(
        df_features,
        df_exp[["clean_formula", "band_gap_exp"]],
        left_on="formula",
        right_on="clean_formula",
        how="inner"
    ).drop_duplicates(subset=["clean_formula"]).reset_index(drop=True)

    print(f"  Обучающая выборка: {len(train_df)} материалов")

    X = train_df[all_feature_cols]
    y = train_df["band_gap_exp"]  # Целевая переменная: лабораторное значение Eg

    # 4. Честная 5-кратная кросс-валидация
    print("  [1/3] Проведение 5-Fold кросс-валидации...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cb_maes, cb_r2s = [], []
    dft_maes, dft_r2s = [], []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Бейзлайн: чистый квантовый расчет DFT без всякого ML
        dft_pred = X_val["band_gap_dft"]
        dft_maes.append(mean_absolute_error(y_val, dft_pred))
        dft_r2s.append(r2_score(y_val, dft_pred))

        # Наша модель CatBoost с регуляризацией от переобучения
        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=4.0,
            verbose=0,
            random_seed=42
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        cb_maes.append(mean_absolute_error(y_val, pred))
        cb_r2s.append(r2_score(y_val, pred))

    mean_dft_mae, mean_dft_r2 = np.mean(dft_maes), np.mean(dft_r2s)
    mean_cb_mae, mean_cb_r2 = np.mean(cb_maes), np.mean(cb_r2s)

    print("\n" + "=" * 65)
    print("  РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ДЛЯ ГЛАВЫ 2 ДИПЛОМА:")
    print("=" * 65)
    print(f"  Бейзлайн: Сырой квантовый расчет (DFT) -> MAE: {mean_dft_mae:.3f} эВ | R²: {mean_dft_r2:.3f}")
    print(f"  Наша модель: CatBoost + Дескрипторы   -> MAE: {mean_cb_mae:.3f} эВ | R²: {mean_cb_r2:.3f}")
    print(f"  Улучшение точности: ошибка снижена на {(1 - mean_cb_mae/mean_dft_mae)*100:.1f}%!")
    print("=" * 65 + "\n")

    # 5. Обучаем финальную модель
    print("  [2/3] Обучение финальной модели на всей выборке...")
    final_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=4.0,
        verbose=0,
        random_seed=42
    )
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    final_model.save_model(model_output_path)

    # 6. Применяем ко всей базе 22 263 материалов
    print("  [3/3] Калибровка всех 22 263 материалов базы...")
    X_full = df_features[all_feature_cols]
    df_features["band_gap_calibrated"] = final_model.predict(X_full).clip(min=0.1)
    df_features["delta_eg_predicted"] = df_features["band_gap_calibrated"] - df_features["band_gap_dft"]

    # Удаляем временную колонку, если была
    if "clean_formula" in df_features.columns:
        df_features = df_features.drop(columns=["clean_formula"])

    os.makedirs(os.path.dirname(output_calibrated_path), exist_ok=True)
    df_features.to_parquet(output_calibrated_path, index=False)
    print(f" Калиброванные данные сохранены в: {output_calibrated_path}")

    # Показываем топ-важных признаков (Feature Importance)
    print("\n Топ-5 признаков, сильнее всего влияющих на поправку:")
    feat_imp = pd.Series(final_model.get_feature_importance(), index=all_feature_cols).sort_values(ascending=False)
    for feat, imp in feat_imp.head(5).items():
        print(f"  • {feat:<35} : {imp:.2f}%")

    return df_features

if __name__ == "__main__":
    train_and_apply_delta_model()