import os
import pandas as pd
from pymatgen.core import Composition
from matminer.datasets import load_dataset

def normalize_formula(formula_or_comp) -> str:
    """
    Приводит любую формулу к канонической редуцированной строке pymatgen.
    Например: 'Ga1As1' -> 'GaAs', 'Ti2O4' -> 'TiO2'
    """
    try:
        if isinstance(formula_or_comp, str):
            comp = Composition(formula_or_comp)
        else:
            comp = formula_or_comp
        return comp.reduced_formula
    except Exception:
        return ""


def load_and_prepare_experimental_dataset(
    cleaned_materials_path: str = "data/02_intermediate/materials_cleaned.parquet",
    output_path: str = "data/04_external/experimental_bandgaps.csv"
) -> pd.DataFrame:
    """
    Загружает академический эталон лабораторных запрещенных зон (matbench_expt_gap),
    сопоставляет его с квантовыми расчетами и формирует обучающую выборку.
    """
    print(" Загрузка эталонных экспериментальных данных...")

    if not os.path.exists(cleaned_materials_path):
        raise FileNotFoundError(
            f"Файл {cleaned_materials_path} не найден! Сначала запусти cleaner.py."
        )

    # 1. Загружаем официальный датасет matbench_expt_gap через matminer
    print("  [1/4] Скачивание/загрузка бенчмарка matbench_expt_gap...")
    try:
        df_expt = load_dataset("matbench_expt_gap")
    except Exception as e:
        print(f"  Ошибка при загрузке matbench_expt_gap: {e}")
        print("  Пробуем альтернативный датасет expt_gap...")
        df_expt = load_dataset("expt_gap")

    print(f"  Успешно загружено лабораторных измерений: {len(df_expt)}")

    # Определяем названия колонок в бенчмарке
    comp_col = "composition" if "composition" in df_expt.columns else "formula"
    target_col = [c for c in df_expt.columns if "gap" in c.lower()][0]

    # 2. Нормализуем формулы лабораторных образцов
    print("  [2/4] Стандартизация химических формул...")
    df_expt["clean_formula"] = df_expt[comp_col].apply(normalize_formula)
    df_expt = df_expt.rename(columns={target_col: "band_gap_exp"})
    df_expt = df_expt.dropna(subset=["clean_formula", "band_gap_exp"])
    df_expt = df_expt[df_expt["clean_formula"] != ""]

    # 3. Загружаем наши материалы из Materials Project
    df_our = pd.read_parquet(cleaned_materials_path)
    df_our["clean_formula"] = df_our["formula"].apply(normalize_formula)

    # 4. Сопоставляем (Inner Join) по стандартизированной формуле
    print("  [3/4] Сопоставление лабораторных экспериментов с квантовыми расчетами...")
    merged = pd.merge(
        df_our[["mp_id", "clean_formula", "band_gap_dft", "density", "crystal_system"]],
        df_expt[["clean_formula", "band_gap_exp"]],
        on="clean_formula",
        how="inner"
    )

    # Удаляем возможные дубликаты совпадений формул
    merged = merged.drop_duplicates(subset=["clean_formula"]).reset_index(drop=True)

    # 5. Вычисляем ключевой таргет для ML: ошибку DFT (Дельту)
    merged["delta_eg"] = merged["band_gap_exp"] - merged["band_gap_dft"]

    # Фильтруем редкие экстремальные артефакты измерения (ошибка больше 5 эВ)
    merged = merged[(merged["delta_eg"] >= -3.0) & (merged["delta_eg"] <= 5.0)]

    # 6. Сохраняем результат
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    
    parquet_path = output_path.replace(".csv", ".parquet")
    merged.to_parquet(parquet_path, index=False)

    print(f"\n Эталонный датасет сформирован и сохранен в: {output_path}")
    print(f" Найдено точных совпадений для обучения ML: {len(merged)} материалов!")
    
    # Научная статистика по ошибке DFT для диплома
    mean_error = merged["delta_eg"].mean()
    underestimation_pct = (merged["delta_eg"] > 0).mean() * 100
    
    print("\n Аналитическая статистика для диплома:")
    print(f"   Средняя величина занижения DFT (PBE): +{mean_error:.2f} эВ")
    print(f"   В {underestimation_pct:.1f}% случаев DFT занижает реальную запрещенную зону!")
    print("\nПримеры сопоставленных материалов (Эксперимент vs DFT):")
    print(merged[["clean_formula", "band_gap_dft", "band_gap_exp", "delta_eg"]].head(10))

    return merged

if __name__ == "__main__":
    load_and_prepare_experimental_dataset()