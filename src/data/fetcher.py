import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

# 1. Загружаем переменные окружения (.env)
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")

def load_config(config_path="configs/data_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_betavoltaic_candidates():
    if not API_KEY or API_KEY == "your_materials_project_api_key_here":
        raise ValueError("Ошибка: Задай валидный MP_API_KEY в файле .env!")

    config = load_config()
    filters = config["screening_filters"]
    fields = config["requested_fields"]

    print("🛰️ Подключение к Materials Project API...")
    print(f" Фильтры: Eg ∈ [{filters['min_band_gap']}, {filters['max_band_gap']}] эВ, "
          f"E_hull <= {filters['max_energy_above_hull']} эВ/атом")

    with MPRester(API_KEY) as mpr:
        # Выполняем поиск по критериям
        docs = mpr.materials.summary.search(
            band_gap=(filters["min_band_gap"], filters["max_band_gap"]),
            energy_above_hull=(0.0, filters["max_energy_above_hull"]),
            fields=fields
        )

    print(f" Загружено {len(docs)} кристаллических структур!")

    # Преобразуем квантовые структуры в чистую плоскую таблицу Pandas
    parsed_records = []
    for doc in docs:
        parsed_records.append({
            "mp_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "band_gap_dft": float(doc.band_gap),
            "density": float(doc.density),
            "volume": float(doc.volume),
            "e_above_hull": float(doc.energy_above_hull),
            "crystal_system": str(doc.symmetry.crystal_system.value),
            "formation_energy": float(doc.formation_energy_per_atom)
        })

    df = pd.DataFrame(parsed_records)

    # Сохраняем в parquet
    output_path = config["output_raw_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f" Данные успешно сохранены в: {output_path}")
    print(f" Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок.")
    print("\nПервые 5 материалов:")
    print(df[["mp_id", "formula", "band_gap_dft", "density", "crystal_system"]].head())

if __name__ == "__main__":
    fetch_betavoltaic_candidates()