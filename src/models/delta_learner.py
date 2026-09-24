import os
import sys
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def train_and_apply_delta_model(
    features_path: str = "data/03_features/materials_features.parquet",
    experimental_path: str = "data/04_external/experimental_bandgaps.parquet",
    model_output_path: str = "models/delta_eg_catboost.cbm",
    output_calibrated_path: str = "data/03_features/materials_calibrated.parquet",
    metrics_output_path: str = "models/metrics_report.json"
):
    print("[*] Запуск модуля машинного обучения (Канонический Delta-Learning, Ramakrishnan et al. 2015)...")

    # 1. Загрузка данных
    df_features = pd.read_parquet(features_path)
    df_exp = pd.read_parquet(experimental_path)

    # 2. Формируем признаки: Magpie + ФУНДАМЕНТАЛЬНАЯ ФИЗИКА (DFT gap, density, volume)
    magpie_cols = [c for c in df_features.columns if "MagpieData" in c]
    physical_cols = ["band_gap_dft", "density", "volume"]
    all_feature_cols = physical_cols + magpie_cols

    print(f"  Признаков: {len(all_feature_cols)} (включая DFT параметры и 132 дескриптора Magpie)")

    # 3. Сопоставляем эксперименты с признаками БЕЗ дублирования колонок
    train_df = pd.merge(
        df_features,
        df_exp[["clean_formula", "band_gap_exp"]],
        left_on="formula",
        right_on="clean_formula",
        how="inner"
    ).drop_duplicates(subset=["clean_formula"]).reset_index(drop=True)

    print(f"  Обучающая выборка: {len(train_df)} материалов")

    X = train_df[all_feature_cols]
    
    # 🎯 КАНОНИЧЕСКИЙ ТАРГЕТ ДЕЛЬТА-ОБУЧЕНИЯ (Ramakrishnan et al., JCTC 2015):
    # Обучаем модель на квантовую систематическую ошибку: Delta_Eg = Eg_exp - Eg_dft
    y_delta = train_df["band_gap_exp"] - train_df["band_gap_dft"]
    y_exp = train_df["band_gap_exp"]
    y_dft = train_df["band_gap_dft"]

    # 4. Честная 5-кратная кросс-валидация
    print("  [1/3] Проведение 5-Fold кросс-валидации для Delta-модели...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    delta_maes, delta_r2s = [], []
    dft_maes, dft_r2s = [], []

    for train_idx, val_idx in kf.split(X, y_delta):
        X_tr, y_tr_delta = X.iloc[train_idx], y_delta.iloc[train_idx]
        X_val, y_val_delta = X.iloc[val_idx], y_delta.iloc[val_idx]
        y_val_actual = y_exp.iloc[val_idx]
        dft_val = y_dft.iloc[val_idx]

        # Бейзлайн: чистый квантовый расчет DFT без всякого ML
        dft_maes.append(mean_absolute_error(y_val_actual, dft_val))
        dft_r2s.append(r2_score(y_val_actual, dft_val))

        # Обучаем модель на дельту Delta_Eg с регуляризацией L2
        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=4.0,
            verbose=0,
            random_seed=42
        )
        model.fit(X_tr, y_tr_delta)
        
        # Калиброванное предсказание по канону: Eg_calibrated = Eg_DFT + Delta_Eg_pred
        pred_delta = model.predict(X_val)
        pred_calibrated = dft_val + pred_delta

        delta_maes.append(mean_absolute_error(y_val_actual, pred_calibrated))
        delta_r2s.append(r2_score(y_val_actual, pred_calibrated))

    mean_dft_mae, mean_dft_r2 = float(np.mean(dft_maes)), float(np.mean(dft_r2s))
    mean_delta_mae, mean_delta_r2 = float(np.mean(delta_maes)), float(np.mean(delta_r2s))
    improvement_pct = float((1.0 - mean_delta_mae / mean_dft_mae) * 100.0)

    print("\n" + "=" * 65)
    print("  РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ДЛЯ ГЛАВЫ 2 ДИПЛОМА (Delta-Learning):")
    print("=" * 65)
    print(f"  Бейзлайн: Сырой квантовый расчет (DFT) -> MAE: {mean_dft_mae:.3f} эВ | R2: {mean_dft_r2:.3f}")
    print(f"  Канонический Delta-Learning (CatBoost)-> MAE: {mean_delta_mae:.3f} эВ | R2: {mean_delta_r2:.3f}")
    print(f"  Улучшение точности: ошибка снижена на {improvement_pct:.1f}%!")
    print("=" * 65 + "\n")

    # 5. Обучаем финальную модель дельты на всей выборке
    print("  [2/3] Обучение финальной Delta-модели на всей экспериментальной выборке...")
    final_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=4.0,
        verbose=0,
        random_seed=42
    )
    final_model.fit(X, y_delta)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    final_model.save_model(model_output_path)

    # Сохраняем отчет о метриках в JSON
    metrics_report = {
        "model_type": "Canonical Delta-Learning (Ramakrishnan et al., 2015)",
        "algorithm": "CatBoostRegressor",
        "features_count": len(all_feature_cols),
        "training_samples": len(train_df),
        "dft_baseline_mae_ev": round(mean_dft_mae, 4),
        "dft_baseline_r2": round(mean_dft_r2, 4),
        "delta_ml_mae_ev": round(mean_delta_mae, 4),
        "delta_ml_r2": round(mean_delta_r2, 4),
        "error_reduction_pct": round(improvement_pct, 2)
    }
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    # 6. Применяем калибровку ко всей базе материалов
    print(f"  [3/3] Калибровка всех {len(df_features)} материалов базы...")
    X_full = df_features[all_feature_cols]
    
    # 🎯 Вычисляем квантовую поправку и калиброванную запрещенную зону:
    df_features["delta_eg_predicted"] = final_model.predict(X_full)
    # Физическое ограничение: калиброванная зона не может быть меньше 0.1 эВ
    df_features["band_gap_calibrated"] = (df_features["band_gap_dft"] + df_features["delta_eg_predicted"]).clip(lower=0.1)

    # Удаляем временную колонку, если была
    if "clean_formula" in df_features.columns:
        df_features = df_features.drop(columns=["clean_formula"])

    os.makedirs(os.path.dirname(output_calibrated_path), exist_ok=True)
    df_features.to_parquet(output_calibrated_path, index=False)
    print(f"[+] Калиброванные данные сохранены в: {output_calibrated_path}")

    # Показываем топ-важных признаков (Feature Importance)
    print("\n Топ-5 признаков, сильнее всего влияющих на квантовую поправку Delta_Eg:")
    feat_imp = pd.Series(final_model.get_feature_importance(), index=all_feature_cols).sort_values(ascending=False)
    for feat, imp in feat_imp.head(5).items():
        print(f"  * {feat:<35} : {imp:.2f}%")

    return df_features

if __name__ == "__main__":
    train_and_apply_delta_model()