"""
Генератор расширенных сравнительных таблиц и графиков/данных для отчета docs/RADIATION_HARDNESS_AUDIT.md
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scripts.audit_radiation_hardness import BENCHMARK_SPECS, ATOMIC_NUMBERS
from src.physics.radiation_transport import (
    ISOTOPE_DECAY_PARAMS,
    mckinley_feshbach_displacement_cross_section,
    compute_spectrum_averaged_displacement_cross_section,
    compute_normalized_fermi_spectrum,
    M_E_C2_KEV, M_U_C2_KEV
)
from src.physics.betavoltaics import (
    calculate_max_recoil_energy,
    calculate_penetration_depth,
    calculate_ehp_energy,
    calculate_theoretical_efficiency
)
from src.database.repository import (
    ELEMENTAL_ECOH, KNOWN_COMPOUND_TM, calculate_cohesive_energy,
    calculate_radiation_displacement_energy
)

def format_float(val, precision=3):
    if val == 0.0:
        return "0.000"
    elif abs(val) < 1e-4:
        return f"{val:.2e}"
    else:
        return f"{val:.{precision}f}"

def generate_detailed_tables():
    isotopes = ["H-3", "Ni-63", "C-14", "Pm-147"]
    
    # Таблица 1: Свойства полупроводников и точность полуэмпирической модели Ed
    t1_rows = []
    for m in BENCHMARK_SPECS:
        f = m["formula"]
        eg = m["Eg_exp_ev"]
        fe = m["delta_Hf_ev"]
        tm = m["Tm_k"]
        ecoh = calculate_cohesive_energy(f, fe)
        
        # Разложение Ed по слагаемым
        term_const = 8.0
        term_eg = 1.8 * eg
        term_ecoh = 2.0 * ecoh
        term_tm = 0.002 * tm
        ed_calc = term_const + term_eg + term_ecoh + term_tm
        
        # Экспериментальное среднее
        subls = m["sublattices"]
        tot_w = sum(s["stoich"] for s in subls)
        ed_exp_mean = sum(s["stoich"] * s["Ed_exp_ev"] for s in subls) / tot_w
        ed_exp_ranges = " / ".join([f"{s['element']}: {s['Ed_exp_range']}" for s in subls])
        
        # Относительная погрешность
        diff_pct = (ed_calc - ed_exp_mean) / ed_exp_mean * 100.0
        
        t1_rows.append({
            "Material": m["material"],
            "Formula": f,
            "Structure": m["crystal_structure"],
            "Eg (eV)": eg,
            "Ecoh (eV)": ecoh,
            "Tm (K)": tm,
            "Ed_model (eV)": ed_calc,
            "Ed_exp (eV)": ed_exp_mean,
            "Ed_ranges (eV)": ed_exp_ranges,
            "Diff (%)": diff_pct,
            "Formula_Breakdown": f"8.0 + {term_eg:.2f} + {term_ecoh:.2f} + {term_tm:.2f}",
            "Ref": m["ref_Ed"]
        })
    df_t1 = pd.DataFrame(t1_rows)
    
    # Таблица 2: Подрешеточная кинематика отдачи T_max (эВ) и пороговые кинетические энергии электронов E_th (кэВ)
    t2_rows = []
    for m in BENCHMARK_SPECS:
        for s in m["sublattices"]:
            elem = s["element"]
            z = s["Z"]
            a = s["A"]
            ed_exp = s["Ed_exp_ev"]
            
            # Порог по энергии электрона E_th
            m_nucl_kev = a * M_U_C2_KEV
            c_const = ed_exp * m_nucl_kev / 2000.0
            e_th_kev = -M_E_C2_KEV + np.sqrt(M_E_C2_KEV**2 + c_const)
            
            row = {
                "Material": m["material"],
                "Sublattice": elem,
                "Z": z,
                "A (amu)": a,
                "Ed_exp (eV)": ed_exp,
                "Eth (keV)": e_th_kev
            }
            for iso in isotopes:
                emax = ISOTOPE_DECAY_PARAMS[iso]["e_max_kev"]
                tmax = calculate_max_recoil_energy(emax, a)
                immune = "✓ (Иммунен)" if tmax < ed_exp else "✗ (Поврежд.)"
                row[f"Tmax_{iso} (eV)"] = tmax
                row[f"Immune_{iso}"] = immune
            t2_rows.append(row)
    df_t2 = pd.DataFrame(t2_rows)
    
    # Таблица 3: Сечения дефектообразования McKinley-Feshbach (на границе E_max и средневзвешенные по спектру Ферми)
    t3_rows = []
    for m in BENCHMARK_SPECS:
        for s in m["sublattices"]:
            elem = s["element"]
            z = s["Z"]
            a = s["A"]
            ed_exp = s["Ed_exp_ev"]
            
            row = {
                "Material": m["material"],
                "Sublattice": elem,
                "Ed_exp (eV)": ed_exp
            }
            for iso in isotopes:
                emax = ISOTOPE_DECAY_PARAMS[iso]["e_max_kev"]
                sig_end = mckinley_feshbach_displacement_cross_section(emax, z, a, ed_exp)
                sig_avg = compute_spectrum_averaged_displacement_cross_section(iso, z, a, ed_exp)
                row[f"sig_end_{iso} (barn)"] = sig_end
                row[f"sig_avg_{iso} (barn)"] = sig_avg
            t3_rows.append(row)
    df_t3 = pd.DataFrame(t3_rows)
    
    return df_t1, df_t2, df_t3

if __name__ == "__main__":
    d1, d2, d3 = generate_detailed_tables()
    print("TABLE 1:")
    print(d1.to_string())
    print("\nTABLE 2:")
    print(d2.to_string())
    print("\nTABLE 3:")
    print(d3.to_string())
