import os
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

def generate_compositional_features(
    input_path: str = "data/02_intermediate/materials_cleaned.parquet",
    output_path: str = "data/03_features/materials_features.parquet"
) -> pd.DataFrame:
    """
    Преобразует химические формулы в 132 физико-химических дескриптора Magpie.
    """
    print(f" Запуск генерации признаков (Featurization) из: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл {input_path} не найден! Сначала запусти cleaner.py.")

    df = pd.read_parquet(input_path)
    print(f"  Загружено материалов: {len(df)}")

    # Шаг 1: Преобразуем строку формулы напрямую через pymatgen (без устаревшего StrToComposition)
    print("  [1/3] Парсинг химических формул в объекты Composition...")
    df["composition_obj"] = df["formula"].apply(lambda f: Composition(f))

    # Шаг 2: Инициализируем классический пресет Magpie
    # Он посчитает статистики (min, max, mean, std) для атомных радиусов,
    # электроотрицательностей, энергий ионизации и валентностей
    print("  [2/3] Вычисление 132 дескрипторов Magpie...")
    ep = ElementProperty.from_preset(preset_name="magpie")
    
    # Считаем признаки (pbar=True покажет красивый прогресс-бар)
    df_features = ep.featurize_dataframe(
        df, 
        col_id="composition_obj", 
        ignore_errors=True,
        pbar=True
    )

    # Шаг 3: Удаляем временный объект pymatgen (он нужен был только для расчета)
    if "composition_obj" in df_features.columns:
        df_features = df_features.drop(columns=["composition_obj"])

    # Шаг 4: Сохраняем готовую матрицу признаков
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    
    print(f"\n Признаки успешно сгенерированы и сохранены в: {output_path}")
    print(f" Итоговый размер матрицы признаков: {df_features.shape[0]} строк на {df_features.shape[1]} колонок!")
    
    # Показываем пример новых колонок
    magpie_cols = [c for c in df_features.columns if "MagpieData" in c]
    print(f"\nПримеры сгенерированных физико-химических признаков (всего {len(magpie_cols)} шт.):")
    for col in magpie_cols[:5]:
        print(f"  • {col}")

    return df_features

if __name__ == "__main__":
    generate_compositional_features()