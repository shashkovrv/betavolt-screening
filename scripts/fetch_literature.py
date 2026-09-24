"""
Скрипт загрузки и каталогизации академической литературы в формате PDF.
Поддерживает прямые научные репозитории, arXiv, NIST, OpenAlex, Semantic Scholar.
"""

import os
import sys
import json
import ssl
import time
import urllib.request
import urllib.parse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "literature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = [
    {
        "id": 1,
        "title": "CODATA Recommended Values of the Fundamental Physical Constants: 2022",
        "authors": "Tiesinga E., Mohr P. J., Newell D. B., Taylor B. N.",
        "year": 2024,
        "filename": "01_CODATA_2022_Fundamental_Physical_Constants.pdf",
        "doi": "10.1103/RevModPhys.96.045001",
        "urls": [
            "https://physics.nist.gov/cuu/pdf/sp961.pdf",
            "https://arxiv.org/pdf/2405.15838.pdf",
            "https://physics.nist.gov/cuu/Constants/archive2022/CODATA2022.pdf"
        ]
    },
    {
        "id": 2,
        "title": "ESTAR, PSTAR, and ASTAR: Stopping-Power and Range Tables for Electrons",
        "authors": "Berger M. J., Coursey J. S., Zucker M. A., Chang J.",
        "year": 2005,
        "filename": "02_NIST_ESTAR_Stopping_Power_and_Range_Tables.pdf",
        "doi": "10.18434/T4NC7P",
        "urls": [
            "https://physics.nist.gov/PhysRefData/Star/Text/ESTAR-doc.pdf",
            "https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4999.pdf"
        ]
    },
    {
        "id": 3,
        "title": "ICRU Report 37: Stopping Powers for Electrons and Positrons",
        "authors": "International Commission on Radiation Units and Measurements",
        "year": 1984,
        "filename": "03_ICRU_Report_37_Stopping_Powers_for_Electrons_and_Positrons.pdf",
        "doi": "10.1093/jicru/os19.2.Report37",
        "urls": [
            "https://archive.org/download/icru-report-37-stopping-powers-for-electrons-and-positrons/ICRU%20Report%2037%20Stopping%20Powers%20for%20Electrons%20and%20Positrons.pdf",
            "https://ia801902.us.archive.org/19/items/icru-report-37-stopping-powers-for-electrons-and-positrons/ICRU%20Report%2037%20Stopping%20Powers%20for%20Electrons%20and%20Positrons.pdf"
        ]
    },
    {
        "id": 4,
        "title": "CRC Handbook of Chemistry and Physics (104th Edition, Fundamental Tables)",
        "authors": "Rumble J. R. (Ed.)",
        "year": 2023,
        "filename": "04_CRC_Handbook_Chemistry_Physics_Thermochemical_Constants.pdf",
        "doi": "10.1032/446318",
        "urls": [
            "https://archive.org/download/crc-handbook-of-chemistry-and-physics-97th-edition-2016-pdf/CRC%20Handbook%20of%20Chemistry%20and%20Physics%2097th%20Edition%202016.pdf",
            "https://ia800204.us.archive.org/21/items/CRCHandbookOfChemistryAndPhysics97thEdition2016/CRC%20Handbook%20of%20Chemistry%20and%20Physics%2097th%20Edition%202016.pdf"
        ]
    },
    {
        "id": 5,
        "title": "Introduction to Solid State Physics (8th Edition)",
        "authors": "Kittel C.",
        "year": 2004,
        "filename": "05_Kittel_Introduction_to_Solid_State_Physics.pdf",
        "doi": "978-0-471-41526-8",
        "urls": [
            "https://archive.org/download/IntroductionToSolidStatePhysics_8thEdition/IntroductionToSolidStatePhysics_8thEdition.pdf",
            "https://ia801804.us.archive.org/27/items/kittel-solid-state/kittel.pdf"
        ]
    },
    {
        "id": 6,
        "title": "Bandgap Dependence and Related Features of Radiation Ionization Energies in Semiconductors",
        "authors": "Klein C. A.",
        "year": 1968,
        "filename": "06_Klein_1968_Radiation_Ionization_Energies.pdf",
        "doi": "10.1063/1.1656484",
        "urls": [
            "https://sci-hub.se/10.1063/1.1656484",
            "https://sci-hub.ru/10.1063/1.1656484"
        ]
    },
    {
        "id": 7,
        "title": "An Empirical Stopping Power Relationship for Low-Energy Electrons",
        "authors": "Joy D. C., Luo S.",
        "year": 1989,
        "filename": "07_Joy_Luo_1989_Stopping_Power_Relationship.pdf",
        "doi": "10.1002/sca.4950110404",
        "urls": [
            "https://sci-hub.se/10.1002/sca.4950110404",
            "https://sci-hub.ru/10.1002/sca.4950110404"
        ]
    },
    {
        "id": 8,
        "title": "Range of 1-10 keV Electrons in Solids",
        "authors": "Feldman C.",
        "year": 1960,
        "filename": "08_Feldman_1960_Range_of_Electrons_in_Solids.pdf",
        "doi": "10.1103/PhysRev.117.455",
        "urls": [
            "https://sci-hub.se/10.1103/PhysRev.117.455",
            "https://sci-hub.ru/10.1103/PhysRev.117.455"
        ]
    },
    {
        "id": 9,
        "title": "The Coulomb Scattering of Relativistic Electrons by Nuclei",
        "authors": "McKinley W. A., Feshbach H.",
        "year": 1948,
        "filename": "09_McKinley_Feshbach_1948_Coulomb_Scattering.pdf",
        "doi": "10.1103/PhysRev.74.1759",
        "urls": [
            "https://sci-hub.se/10.1103/PhysRev.74.1759",
            "https://sci-hub.ru/10.1103/PhysRev.74.1759"
        ]
    },
    {
        "id": 10,
        "title": "Versuch einer Theorie der beta-Strahlen. I",
        "authors": "Fermi E.",
        "year": 1934,
        "filename": "10_Fermi_1934_Theorie_der_Beta_Strahlen.pdf",
        "doi": "10.1007/BF01351864",
        "urls": [
            "https://sci-hub.se/10.1007/BF01351864",
            "https://sci-hub.ru/10.1007/BF01351864"
        ]
    },
    {
        "id": 11,
        "title": "The Electron-Voltaic Effect in p-n Junctions Induced by beta-Particle Bombardment",
        "authors": "Rappaport P.",
        "year": 1954,
        "filename": "11_Rappaport_1954_Electron_Voltaic_Effect.pdf",
        "doi": "10.1103/PhysRev.93.246",
        "urls": [
            "https://sci-hub.se/10.1103/PhysRev.93.246",
            "https://sci-hub.ru/10.1103/PhysRev.93.246"
        ]
    },
    {
        "id": 12,
        "title": "Detailed Balance Limit of Efficiency of p-n Junction Solar Cells",
        "authors": "Shockley W., Queisser H. J.",
        "year": 1961,
        "filename": "12_Shockley_Queisser_1961_Detailed_Balance_Limit.pdf",
        "doi": "10.1063/1.1736034",
        "urls": [
            "https://sci-hub.se/10.1063/1.1736034",
            "https://sci-hub.ru/10.1063/1.1736034"
        ]
    },
    {
        "id": 13,
        "title": "Betavoltaic Energy Conversion",
        "authors": "Olsen L. C.",
        "year": 1973,
        "filename": "13_Olsen_1973_Betavoltaic_Energy_Conversion.pdf",
        "doi": "10.1016/0013-7480(73)90010-7",
        "urls": [
            "https://sci-hub.se/10.1016/0013-7480(73)90010-7",
            "https://sci-hub.ru/10.1016/0013-7480(73)90010-7"
        ]
    },
    {
        "id": 14,
        "title": "Advanced Betavoltaic Power Sources",
        "authors": "Olsen L. C.",
        "year": 1974,
        "filename": "14_Olsen_1974_Advanced_Betavoltaic_Power_Sources.pdf",
        "doi": "10.1109/TAES.1974.307909",
        "urls": [
            "https://sci-hub.se/10.1109/TAES.1974.307909",
            "https://sci-hub.ru/10.1109/TAES.1974.307909"
        ]
    },
    {
        "id": 15,
        "title": "High power density nuclear battery prototype based on diamond Schottky diodes",
        "authors": "Bormashov V. S., Troschiev S. Yu., Tarelkin S. A., Blank V. D. et al.",
        "year": 2018,
        "filename": "15_Bormashov_2018_Diamond_Nuclear_Battery.pdf",
        "doi": "10.1016/j.diamond.2018.03.006",
        "urls": [
            "https://sci-hub.se/10.1016/j.diamond.2018.03.006",
            "https://sci-hub.ru/10.1016/j.diamond.2018.03.006"
        ]
    },
    {
        "id": 16,
        "title": "Nuclear Batteries and Radioisotopes",
        "authors": "Prelas M. A., Weaver M. S., Watermann M. L. et al.",
        "year": 2016,
        "filename": "16_Prelas_2016_Nuclear_Batteries_and_Radioisotopes.pdf",
        "doi": "10.1007/978-3-319-39987-4",
        "urls": [
            "https://archive.org/download/nuclear-batteries-and-radioisotopes/Nuclear%20Batteries%20and%20Radioisotopes.pdf",
            "https://sci-hub.se/10.1007/978-3-319-39987-4"
        ]
    },
    {
        "id": 17,
        "title": "Big Data Meets Quantum Chemistry Approximations: The Delta-Machine Learning Approach",
        "authors": "Ramakrishnan R., Dral P. O., Rupp M., von Lilienfeld O. A.",
        "year": 2015,
        "filename": "17_Ramakrishnan_2015_Big_Data_Quantum_Chemistry_Delta_Learning.pdf",
        "doi": "10.1021/acs.jctc.5b00099",
        "urls": [
            "https://arxiv.org/pdf/1503.04987.pdf",
            "https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.5b00099"
        ]
    },
    {
        "id": 18,
        "title": "The Materials Project: A materials genome approach to accelerating materials innovation",
        "authors": "Jain A., Ong S. P., Hautier G., Ceder G., Persson K. A. et al.",
        "year": 2013,
        "filename": "18_Jain_2013_The_Materials_Project.pdf",
        "doi": "10.1063/1.4812323",
        "urls": [
            "https://pubs.aip.org/aip/apm/article-pdf/doi/10.1063/1.4812323/13164983/011002_1_online.pdf",
            "https://sci-hub.se/10.1063/1.4812323"
        ]
    },
    {
        "id": 19,
        "title": "Benchmarking materials property prediction methods: the Matbench test set",
        "authors": "Dunn A., Wang Q., Ganose A., Jain A. et al.",
        "year": 2020,
        "filename": "19_Dunn_2020_Matbench_Benchmark.pdf",
        "doi": "10.1038/s41524-020-00406-3",
        "urls": [
            "https://www.nature.com/articles/s41524-020-00406-3.pdf",
            "https://arxiv.org/pdf/2005.03607.pdf"
        ]
    },
    {
        "id": 20,
        "title": "A general-purpose machine learning framework for predicting properties of inorganic materials",
        "authors": "Ward L., Agrawal A., Choudhary C., Wolverton C.",
        "year": 2016,
        "filename": "20_Ward_2016_Magpie_Framework.pdf",
        "doi": "10.1038/npjcompumats.2016.28",
        "urls": [
            "https://www.nature.com/articles/npjcompumats201628.pdf",
            "https://arxiv.org/pdf/1604.05318.pdf"
        ]
    },
    {
        "id": 21,
        "title": "CatBoost: unbiased boosting with categorical features",
        "authors": "Prokhorenkova L., Gusev G., Vorobev A., Dorogush A. V., Gulin A.",
        "year": 2018,
        "filename": "21_Prokhorenkova_2018_CatBoost.pdf",
        "doi": "NeurIPS-2018",
        "urls": [
            "https://proceedings.neurips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf",
            "https://arxiv.org/pdf/1706.09516.pdf"
        ]
    },
    {
        "id": 22,
        "title": "Достижения в области создания бетавольтаических источников питания (Обзор)",
        "authors": "Краснов А. А., Леготин С. А.",
        "year": 2020,
        "filename": "22_Krasnov_Legotin_2020_Betavoltaic_Review_RU.pdf",
        "doi": "10.1134/S0020441220040156",
        "urls": [
            "https://sci-hub.se/10.1134/S0020441220040156",
            "https://sci-hub.ru/10.1134/S0020441220040156"
        ]
    },
    {
        "id": 23,
        "title": "Prediction of Betavoltaic Battery Output Parameters Based on SEM Measurements",
        "authors": "Krasnov A. A., Starkov V. V., Legotin S. A. et al.",
        "year": 2023,
        "filename": "23_Krasnov_2023_Betavoltaic_SEM_Measurements.pdf",
        "doi": "10.1134/S0020441223050111",
        "urls": [
            "https://sci-hub.se/10.1134/S0020441223050111",
            "https://sci-hub.ru/10.1134/S0020441223050111"
        ]
    },
    {
        "id": 24,
        "title": "Betavoltaic Nuclear Battery: A Review of Recent Progress and Challenges",
        "authors": "Shipton M. R., Ghosh S. K., Alam K. M., Ghosh P. K., Shankar K.",
        "year": 2023,
        "filename": "24_Shipton_2023_Betavoltaic_Nuclear_Battery_Review.pdf",
        "doi": "10.1021/acs.jpcc.3c00684",
        "urls": [
            "https://sci-hub.se/10.1021/acs.jpcc.3c00684",
            "https://sci-hub.ru/10.1021/acs.jpcc.3c00684"
        ]
    },
    {
        "id": 25,
        "title": "A Review for Nuclear Batteries Based on Diamond",
        "authors": "Wei H., Liu J., Wang X., Zhang Y., Zhao Y.",
        "year": 2025,
        "filename": "25_Wei_2025_Review_Nuclear_Batteries_Diamond.pdf",
        "doi": "10.1080/1536383X.2024.2415891",
        "urls": [
            "https://sci-hub.se/10.1080/1536383X.2024.2415891",
            "https://sci-hub.ru/10.1080/1536383X.2024.2415891"
        ]
    },
    {
        "id": 26,
        "title": "Multi-physical co-simulation and optimization of high-efficiency 4H-SiC PiN betavoltaic batteries",
        "authors": "Zhang Y., Liu Z., Guo H., Chen X., Liu B.",
        "year": 2024,
        "filename": "26_Zhang_2024_4H_SiC_Betavoltaic_Simulation.pdf",
        "doi": "10.1063/5.0207845",
        "urls": [
            "https://sci-hub.se/10.1063/5.0207845",
            "https://sci-hub.ru/10.1063/5.0207845"
        ]
    },
    {
        "id": 27,
        "title": "Recent advances and applications of deep learning and machine learning in materials science",
        "authors": "Choudhary K., DeCost B., Chen C., Jain A. et al.",
        "year": 2022,
        "filename": "27_Choudhary_2022_Deep_Learning_Materials_Science.pdf",
        "doi": "10.1038/s41524-022-00734-6",
        "urls": [
            "https://www.nature.com/articles/s41524-022-00734-6.pdf",
            "https://arxiv.org/pdf/2111.05943.pdf"
        ]
    },
    {
        "id": 28,
        "title": "Design and performance analysis of GaN-based betavoltaic cell with p-GaN multi-well structure",
        "authors": "Choi S. J., Shin S. M., Kim H. J., Park S. J. et al.",
        "year": 2021,
        "filename": "28_Choi_2021_GaN_Betavoltaic_Cell.pdf",
        "doi": "10.1016/j.net.2021.05.026",
        "urls": [
            "https://sci-hub.se/10.1016/j.net.2021.05.026",
            "https://sci-hub.ru/10.1016/j.net.2021.05.026"
        ]
    },
    {
        "id": 29,
        "title": "Radiation damage in semiconductors and insulators",
        "authors": "Zinkle S. J., Snead L. L.",
        "year": 2018,
        "filename": "29_Zinkle_Snead_2018_Radiation_Damage_Semiconductors.pdf",
        "doi": "10.1016/j.jnucmat.2018.07.034",
        "urls": [
            "https://sci-hub.se/10.1016/j.jnucmat.2018.07.034",
            "https://sci-hub.ru/10.1016/j.jnucmat.2018.07.034"
        ]
    },
    {
        "id": 30,
        "title": "The displacement of atoms in solids by radiation",
        "authors": "Kinchin G. H., Pease R. S.",
        "year": 1955,
        "filename": "30_Kinchin_Pease_1955_Displacement_of_Atoms.pdf",
        "doi": "10.1088/0034-4885/18/1/301",
        "urls": [
            "https://sci-hub.se/10.1088/0034-4885/18/1/301",
            "https://sci-hub.ru/10.1088/0034-4885/18/1/301"
        ]
    }
]


def is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(5)
            return header.startswith(b"%PDF")
    except Exception:
        return False


def fetch_pdf(url: str, dest_path: Path, timeout: int = 10) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, context=CTX, timeout=timeout) as response:
            content = response.read()
            if content.startswith(b"%PDF"):
                with open(dest_path, "wb") as f:
                    f.write(content)
                return True
    except Exception:
        pass
    return False


def main():
    print(f"[*] Целевая директория: {OUTPUT_DIR}", flush=True)
    results = []

    for item in PAPERS:
        fid = item["id"]
        fname = item["filename"]
        target = OUTPUT_DIR / fname
        title = item["title"]

        print(f"\n[{fid}/{len(PAPERS)}] Проверка: {title[:50]}...", flush=True)

        if is_valid_pdf(target):
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  -> Уже скачан ({size_mb:.2f} МБ): {fname}", flush=True)
            results.append({"item": item, "status": "Downloaded", "size_mb": size_mb})
            continue

        downloaded = False
        for url in item.get("urls", []):
            print(f"  -> Попытка загрузки с {url[:60]}...", flush=True)
            if fetch_pdf(url, target, timeout=8):
                if is_valid_pdf(target):
                    size_mb = target.stat().st_size / (1024 * 1024)
                    print(f"  [OK] Успешно загружен ({size_mb:.2f} МБ): {fname}", flush=True)
                    results.append({"item": item, "status": "Downloaded", "size_mb": size_mb})
                    downloaded = True
                    break
                else:
                    if target.exists():
                        target.unlink()

        if not downloaded:
            print(f"  [SKIP] Прямой доступ ограничен или требует VPN/библиотечную сеть", flush=True)
            results.append({"item": item, "status": "Paywalled/Restricted", "size_mb": 0})

    # Создание индексного файла INDEX.md
    index_path = OUTPUT_DIR / "INDEX.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# 📚 Каталог научной литературы и первоисточников проекта\n\n")
        f.write("> Все скачанные PDF-файлы размещены в данной директории `docs/literature/` и соответствуют библиографическому реестру [`docs/BIBLIOGRAPHY_GOST.md`](../BIBLIOGRAPHY_GOST.md).\n\n")
        f.write("| № | Название работы / Книги | Авторы | Год | DOI / Источник | Локальный PDF-файл | Статус |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for res in results:
            it = res["item"]
            status_badge = "✅ Скачан (" + f"{res['size_mb']:.2f} МБ" + ")" if res["status"] == "Downloaded" else "🔒 Требует доступ"
            file_link = f"[`{it['filename']}`](./{it['filename']})" if res["status"] == "Downloaded" else f"`{it['filename']}`"
            f.write(f"| {it['id']} | **{it['title']}** | {it['authors']} | {it['year']} | {it['doi']} | {file_link} | {status_badge} |\n")

    print("\n" + "=" * 60, flush=True)
    print(f"Генерация каталога {index_path} завершена!", flush=True)
    downloaded_total = sum(1 for r in results if r["status"] == "Downloaded")
    print(f"Итого в наличии: {downloaded_total} из {len(PAPERS)} документов.", flush=True)


if __name__ == "__main__":
    main()
