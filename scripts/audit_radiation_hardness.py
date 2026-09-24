"""
Скрипт прецизионного сравнительного анализа радиационной стойкости:
- Полуэмпирическая модель E_d = 8.0 + 1.8*E_g + 2.0*E_coh + 0.002*T_m
- Релятивистская кинематика отдачи T_max (средневзвешенная vs по подрешеткам)
- Сечения дефектообразования McKinley-Feshbach для 4 изотопов (Ni-63, H-3, C-14, Pm-147)
- Сопоставление с экспериментальными данными литературы
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import sqlite3
from src.physics.betavoltaics import (
    ISOTOPES, ATOMIC_WEIGHTS, calculate_max_recoil_energy,
    calculate_penetration_depth, calculate_ehp_energy, calculate_theoretical_efficiency
)
from src.physics.radiation_transport import (
    ISOTOPE_DECAY_PARAMS,
    mckinley_feshbach_displacement_cross_section,
    compute_spectrum_averaged_displacement_cross_section,
    compute_normalized_fermi_spectrum,
    M_E_C2_KEV, M_U_C2_KEV
)
from src.database.repository import (
    ELEMENTAL_ECOH, KNOWN_COMPOUND_TM, calculate_cohesive_energy,
    calculate_radiation_displacement_energy, parse_composition
)

# Атомные номера элементов
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ti': 22, 'Ga': 31, 'As': 33
}

# Эталонные полупроводники для аудита
BENCHMARK_SPECS = [
    {
        "material": "Si",
        "name": "Кремний",
        "mp_id": "mp-149",
        "formula": "Si",
        "crystal_structure": "Diamond cubic (Fd-3m)",
        "density_g_cm3": 2.33,
        "Eg_exp_ev": 1.12,
        "delta_Hf_ev": 0.0,
        "Tm_k": 1687.0,
        "sublattices": [
            {"element": "Si", "Z": 14, "A": 28.085, "stoich": 1.0, "Ed_exp_ev": 13.0, "Ed_exp_range": "13 - 21"}
        ],
        "ref_Ed": "Loferski & Rappaport (1958), Corbett (1966), ASTM E521"
    },
    {
        "material": "Diamond C",
        "name": "Алмаз",
        "mp_id": "mp-66",
        "formula": "C",
        "crystal_structure": "Diamond cubic (Fd-3m)",
        "density_g_cm3": 3.51,
        "Eg_exp_ev": 5.47,
        "delta_Hf_ev": 0.0,
        "Tm_k": 3800.0,
        "sublattices": [
            {"element": "C", "Z": 6, "A": 12.011, "stoich": 1.0, "Ed_exp_ev": 40.0, "Ed_exp_range": "37.5 - 45"}
        ],
        "ref_Ed": "Clark et al. (1952), Koike et al. (1992), Campbell (2007)"
    },
    {
        "material": "4H-SiC",
        "name": "Карбид кремния 4H",
        "mp_id": "mp-1204356",
        "formula": "SiC",
        "crystal_structure": "Hexagonal (P6_3mc)",
        "density_g_cm3": 3.21,
        "Eg_exp_ev": 3.25,  # 4H-SiC exp gap
        "delta_Hf_ev": -0.34,
        "Tm_k": 3100.0,
        "sublattices": [
            {"element": "C", "Z": 6, "A": 12.011, "stoich": 0.5, "Ed_exp_ev": 20.0, "Ed_exp_range": "19 - 22"},
            {"element": "Si", "Z": 14, "A": 28.085, "stoich": 0.5, "Ed_exp_ev": 35.0, "Ed_exp_range": "32 - 38"}
        ],
        "ref_Ed": "Steeds et al. (2002), Weber et al. (2004), Zinkle & Kinoshita (1997)"
    },
    {
        "material": "GaN",
        "name": "Нитрид галлия (вюрцит)",
        "mp_id": "mp-804",
        "formula": "GaN",
        "crystal_structure": "Wurtzite (P6_3mc)",
        "density_g_cm3": 6.15,
        "Eg_exp_ev": 3.42,
        "delta_Hf_ev": -0.60,
        "Tm_k": 2773.0,
        "sublattices": [
            {"element": "N", "Z": 7, "A": 14.007, "stoich": 0.5, "Ed_exp_ev": 20.5, "Ed_exp_range": "19 - 22"},
            {"element": "Ga", "Z": 31, "A": 69.723, "stoich": 0.5, "Ed_exp_ev": 25.0, "Ed_exp_range": "22 - 28"}
        ],
        "ref_Ed": "Look et al. (1997, 2001), Emtsev et al. (2001)"
    },
    {
        "material": "TiO2 (Рутил)",
        "name": "Диоксид титана (рутил)",
        "mp_id": "mp-2657",
        "formula": "TiO2",
        "crystal_structure": "Rutile (P4_2/mnm)",
        "density_g_cm3": 4.23,
        "Eg_exp_ev": 3.03,
        "delta_Hf_ev": -3.20,
        "Tm_k": 2116.0,
        "sublattices": [
            {"element": "O", "Z": 8, "A": 15.999, "stoich": 2.0/3.0, "Ed_exp_ev": 25.0, "Ed_exp_range": "23 - 30"},
            {"element": "Ti", "Z": 22, "A": 47.867, "stoich": 1.0/3.0, "Ed_exp_ev": 48.0, "Ed_exp_range": "42 - 52"}
        ],
        "ref_Ed": "Robinson (1974), Thomas et al. (2017), Zheng et al. (2018)"
    },
    {
        "material": "c-BN",
        "name": "Кубический нитрид бора",
        "mp_id": "mp-604884",  # or mp-1639
        "formula": "BN",
        "crystal_structure": "Zincblende (F-43m)",
        "density_g_cm3": 3.45,
        "Eg_exp_ev": 6.20,
        "delta_Hf_ev": -1.35,
        "Tm_k": 3246.0,
        "sublattices": [
            {"element": "B", "Z": 5, "A": 10.81, "stoich": 0.5, "Ed_exp_ev": 20.0, "Ed_exp_range": "18 - 22"},
            {"element": "N", "Z": 7, "A": 14.007, "stoich": 0.5, "Ed_exp_ev": 25.0, "Ed_exp_range": "22 - 30"}
        ],
        "ref_Ed": "Zubavichus et al. (2004), Kotakoski et al. (2010), Jin et al. (2009)"
    },
    {
        "material": "BP",
        "name": "Фосфид бора",
        "mp_id": "mp-1479",
        "formula": "BP",
        "crystal_structure": "Zincblende (F-43m)",
        "density_g_cm3": 2.97,
        "Eg_exp_ev": 2.02,
        "delta_Hf_ev": -0.60,
        "Tm_k": 2270.0,
        "sublattices": [
            {"element": "B", "Z": 5, "A": 10.81, "stoich": 0.5, "Ed_exp_ev": 18.0, "Ed_exp_range": "15 - 20"},
            {"element": "P", "Z": 15, "A": 30.974, "stoich": 0.5, "Ed_exp_ev": 24.0, "Ed_exp_range": "20 - 26"}
        ],
        "ref_Ed": "Vetter et al. (2003), Paderno et al. (2007)"
    },
    {
        "material": "AlN",
        "name": "Нитрид алюминия",
        "mp_id": "mp-661",
        "formula": "AlN",
        "crystal_structure": "Wurtzite (P6_3mc)",
        "density_g_cm3": 3.26,
        "Eg_exp_ev": 6.13,
        "delta_Hf_ev": -1.65,
        "Tm_k": 2470.0,
        "sublattices": [
            {"element": "N", "Z": 7, "A": 14.007, "stoich": 0.5, "Ed_exp_ev": 22.0, "Ed_exp_range": "20 - 25"},
            {"element": "Al", "Z": 13, "A": 26.982, "stoich": 0.5, "Ed_exp_ev": 32.0, "Ed_exp_range": "28 - 36"}
        ],
        "ref_Ed": "Vurgaftman & Meyer (2003), Yan et al. (2014)"
    },
    {
        "material": "B4C",
        "name": "Карбид бора",
        "mp_id": "mp-1238815",
        "formula": "B4C",
        "crystal_structure": "Rhombohedral (R-3m)",
        "density_g_cm3": 2.52,
        "Eg_exp_ev": 2.09,
        "delta_Hf_ev": -0.15,
        "Tm_k": 2720.0,
        "sublattices": [
            {"element": "B", "Z": 5, "A": 10.81, "stoich": 0.8, "Ed_exp_ev": 20.0, "Ed_exp_range": "18 - 24"},
            {"element": "C", "Z": 6, "A": 12.011, "stoich": 0.2, "Ed_exp_ev": 28.0, "Ed_exp_range": "25 - 32"}
        ],
        "ref_Ed": "Emin (2006), Domnich et al. (2011), Gosset et al. (2008)"
    }
]


def run_full_audit():
    print("=" * 110)
    print("  ПРЕЦИЗИОННЫЙ АУДИТ ПОЛУЭМПИРИЧЕСКОЙ МОДЕЛИ E_d, КИНЕМАТИКИ ОТДАЧИ И СЕЧЕНИЙ ДЕФЕКТООБРАЗОВАНИЯ")
    print("=" * 110)
    
    audit_results = []
    sublattice_audit_results = []
    cross_section_results = []
    
    isotopes = ["H-3", "Ni-63", "C-14", "Pm-147"]
    
    for mat in BENCHMARK_SPECS:
        formula = mat["formula"]
        eg = mat["Eg_exp_ev"]
        fe = mat["delta_Hf_ev"]
        tm = mat["Tm_k"]
        
        # 1. Расчет энергии когезии и полуэмпирической Ed
        ecoh = calculate_cohesive_energy(formula, fe)
        ed_model = calculate_radiation_displacement_energy(eg, ecoh, tm)
        
        # Эффективная средняя масса формульной единицы
        total_atoms = sum(s["stoich"] for s in mat["sublattices"])
        a_eff = sum(s["stoich"] * s["A"] for s in mat["sublattices"]) / total_atoms
        z_eff = sum(s["stoich"] * s["Z"] for s in mat["sublattices"]) / total_atoms
        
        # Средняя экспериментальная Ed
        ed_exp_weighted = sum(s["stoich"] * s["Ed_exp_ev"] for s in mat["sublattices"]) / total_atoms
        
        rel_diff_pct = (ed_model - ed_exp_weighted) / ed_exp_weighted * 100.0
        
        res_mat = {
            "Material": mat["material"],
            "Formula": formula,
            "Structure": mat["crystal_structure"],
            "Eg_eV": eg,
            "Ecoh_eV_atom": ecoh,
            "Tm_K": tm,
            "Ed_model_eV": ed_model,
            "Ed_exp_mean_eV": ed_exp_weighted,
            "Diff_Ed_pct": rel_diff_pct,
            "A_eff": a_eff,
            "Z_eff": z_eff,
            "Ref": mat["ref_Ed"]
        }
        audit_results.append(res_mat)
        
        # 2. Кинематика лобового соударения T_max для эффективной массы и по подрешеткам
        for iso_key in isotopes:
            e_max = ISOTOPE_DECAY_PARAMS[iso_key]["e_max_kev"]
            t_max_eff = calculate_max_recoil_energy(e_max, a_eff)
            res_mat[f"Tmax_{iso_key}_eff_eV"] = t_max_eff
            res_mat[f"Immune_{iso_key}_eff_model"] = (t_max_eff < ed_model)
            res_mat[f"Immune_{iso_key}_eff_exp"] = (t_max_eff < ed_exp_weighted)
            
            # Сечение Мак-Кинли-Фешбаха на эффективном атоме
            sig_eff_endpoint = mckinley_feshbach_displacement_cross_section(e_max, z_eff, a_eff, ed_model)
            sig_eff_avg = compute_spectrum_averaged_displacement_cross_section(iso_key, z_eff, a_eff, ed_model)
            res_mat[f"Sigma_endpoint_{iso_key}_eff_barn"] = sig_eff_endpoint
            res_mat[f"Sigma_avg_{iso_key}_eff_barn"] = sig_eff_avg
        
        # 3. Детальный аудит подрешеток (Sublattice-resolved analysis)
        for subl in mat["sublattices"]:
            elem = subl["element"]
            z_elem = subl["Z"]
            a_elem = subl["A"]
            ed_subl_exp = subl["Ed_exp_ev"]
            
            subl_row = {
                "Material": mat["material"],
                "Sublattice": elem,
                "Z": z_elem,
                "A": a_elem,
                "Ed_subl_exp_eV": ed_subl_exp,
                "Ed_subl_range": subl["Ed_exp_range"],
                "Ed_compound_model_eV": ed_model
            }
            
            for iso_key in isotopes:
                e_max = ISOTOPE_DECAY_PARAMS[iso_key]["e_max_kev"]
                t_max_subl = calculate_max_recoil_energy(e_max, a_elem)
                
                # Пороговая кинетическая энергия электрона для смещения в данной подрешетке
                # Решение уравнения T_max(E_th) = Ed:
                # E_th * (E_th + 2*m_e*c^2) = Ed * M_nucleus * c^2 / 2000
                m_nucl_kev = a_elem * M_U_C2_KEV
                c_const = ed_subl_exp * m_nucl_kev / 2000.0
                # E_th^2 + 2*m_e*c^2 * E_th - c = 0
                # E_th = -m_e*c^2 + sqrt((m_e*c^2)^2 + c)
                e_thresh_kev = -M_E_C2_KEV + np.sqrt(M_E_C2_KEV**2 + c_const)
                
                # Сечение дефектообразования для данной подрешетки по экспериментальному Ed
                sig_endpoint_exp = mckinley_feshbach_displacement_cross_section(e_max, z_elem, a_elem, ed_subl_exp)
                sig_avg_exp = compute_spectrum_averaged_displacement_cross_section(iso_key, z_elem, a_elem, ed_subl_exp)
                
                # Сечение дефектообразования для данной подрешетки по моделированному Ed
                sig_endpoint_mod = mckinley_feshbach_displacement_cross_section(e_max, z_elem, a_elem, ed_model)
                sig_avg_mod = compute_spectrum_averaged_displacement_cross_section(iso_key, z_elem, a_elem, ed_model)
                
                subl_row[f"Tmax_{iso_key}_eV"] = t_max_subl
                subl_row[f"Eth_{iso_key}_keV"] = e_thresh_kev
                subl_row[f"Immune_{iso_key}_exp"] = (t_max_subl < ed_subl_exp)
                subl_row[f"Sigma_endpoint_{iso_key}_barn"] = sig_endpoint_exp
                subl_row[f"Sigma_avg_{iso_key}_barn"] = sig_avg_exp
                subl_row[f"Sigma_avg_model_{iso_key}_barn"] = sig_avg_mod
            
            sublattice_audit_results.append(subl_row)
            
    df_audit = pd.DataFrame(audit_results)
    df_subl = pd.DataFrame(sublattice_audit_results)
    
    print("\n[1] СОПОСТАВЛЕНИЕ ПОЛУЭМПИРИЧЕСКОЙ МОДЕЛИ E_d С ЭКСПЕРИМЕНТОМ:")
    cols_display = ["Material", "Eg_eV", "Ecoh_eV_atom", "Tm_K", "Ed_model_eV", "Ed_exp_mean_eV", "Diff_Ed_pct"]
    print(df_audit[cols_display].to_string(index=False))
    
    print("\n[2] РЕЛЯТИВИСТСКАЯ КИНЕМАТИКА T_max ПО ПОДРЕШЕТКАМ ДЛЯ 4 ИЗОТОПОВ (в эВ):")
    cols_kinem = ["Material", "Sublattice", "Ed_subl_exp_eV", "Tmax_H-3_eV", "Tmax_Ni-63_eV", "Tmax_C-14_eV", "Tmax_Pm-147_eV"]
    print(df_subl[cols_kinem].to_string(index=False))
    
    print("\n[3] ПОРОГОВЫЕ ЭНЕРГИИ ЭЛЕКТРОНОВ E_th (кэВ) И СЕЧЕНИЯ ДЕФЕКТООБРАЗОВАНИЯ <sigma_d> (барн):")
    cols_sig = ["Material", "Sublattice", "Eth_H-3_keV", "Sigma_avg_Ni-63_barn", "Sigma_avg_C-14_barn", "Sigma_avg_Pm-147_barn"]
    print(df_subl[cols_sig].to_string(index=False))
    
    return df_audit, df_subl

if __name__ == "__main__":
    df_a, df_s = run_full_audit()
