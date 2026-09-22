import os
import pandas as pd
from pymatgen.core import Composition

# 1. Формируем "черный список" элементов
# Радиоактивные актиниды, лантаноиды без стабильных изотопов, благородные газы и токсичные металлы
FORBIDDEN_ELEMENTS = {
    # Радиоактивные / трансурановые
    "Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", 
    "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    # Благородные газы (не образуют стабильных полупроводников при н.у.)
    "He", "Ne", "Ar", "Kr", "Xe",
    # Чрезвычайно токсичные металлы с высоким давлением пара
    "Hg",  # Ртуть (жидкая/пары, разрушает электронику)
    "Tl",  # Таллий (смертельный кумулятивный яд)
}


def contains_forbidden_elements(formula: str) -> bool:
    """
    Проверяет, содержит ли химическая формула элементы из черного списка.
    Использует pymatgen для корректного химического парсинга.
    """
    try:
        comp = Composition(formula)
        elements_in_material = {el.symbol for el in comp.elements}
        # Если есть пересечение множеств — возвращаем True (материал запрещен)
        return len(elements_in_material.intersection(FORBIDDEN_ELEMENTS)) > 0
    except Exception:
        # Если формула повреждена или не парсится — отсекаем
        return True


def clean_materials_data(
    input_path: str = "data/01_raw/materials_raw.parquet",
    output_path: str = "data/02_intermediate/materials_cleaned.parquet"
) -> pd.DataFrame:
    """
    Основной пайплайн очистки датасета.
    """
    print(f" Запуск очистки данных из: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл {input_path} не найден! Сначала запусти fetcher.py.")

    df = pd.read_parquet(input_path)
    initial_count = len(df)
    print(f"  Исходное количество записей: {initial_count}")

    # --- Шаг 1: Санитарный фильтр (отсутствующие и нефизичные значения) ---
    df = df.dropna(subset=["formula", "band_gap_dft", "density", "e_above_hull"])
    df = df[(df["density"] > 0) & (df["volume"] > 0) & (df["band_gap_dft"] > 0)]
    after_sanity = len(df)
    print(f"  После проверки физической корректности: {after_sanity} (удалено: {initial_count - after_sanity})")

    # --- Шаг 2: Фильтрация по черному списку элементов ---
    # Применяем проверку ко всем формулам
    forbidden_mask = df["formula"].apply(contains_forbidden_elements)
    df = df[~forbidden_mask].reset_index(drop=True)
    after_elements = len(df)
    print(f"  После удаления радиоактивных/токсичных элементов: {after_elements} (удалено: {after_sanity - after_elements})")

    # --- Шаг 3: Разрешение полиморфизма (дедупликация по формуле) ---
    # Сортируем по e_above_hull (самые стабильные кристаллы идут первыми)
    df = df.sort_values(by="e_above_hull", ascending=True)
    
    # Оставляем только самую стабильную кристаллическую модификацию для каждой формулы
    df = df.drop_duplicates(subset=["formula"], keep="first").reset_index(drop=True)
    after_dedup = len(df)
    print(f"  После выбора наиболее стабильных полиморфов: {after_dedup} (удалено дубликатов: {after_elements - after_dedup})")

    # --- Шаг 4: Сохранение результата ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f" Очищенный датасет сохранен в: {output_path}")
    
    # Показываем красивый отчет
    print("\n Первые 5 очищенных материалов (теперь без Актиния!):")
    print(df[["mp_id", "formula", "band_gap_dft", "density", "e_above_hull", "crystal_system"]].head())

    return df


if __name__ == "__main__":
    clean_materials_data()