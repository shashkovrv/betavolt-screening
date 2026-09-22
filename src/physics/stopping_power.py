import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def joy_luo_stopping_power(energy_kev: np.ndarray, density: float, z_eff: float, a_eff: float) -> np.ndarray:
    """
    Тормозная способность электронов dE/dx (в кэВ / мкм).
    Использует общепринятую модификацию Бете-Блоха Джоя-Ло (Joy & Luo) для энергий 1-100 кэВ.
    """
    E = np.maximum(energy_kev, 0.5)
    
    # Средний потенциал ионизации J (в кэВ) с поправкой Джоя-Ло
    J = (11.5 * z_eff) * 1e-3  # кэВ
    k = 0.73  # полуэмпирический коэффициент связи
    
    # Эффективный логарифмический аргумент
    argument = 1.166 * (E + k * J) / J
    argument = np.maximum(argument, 1.01)
    
    # dE/dx в кэВ / мкм
    # Префактор: 78.5 * (rho * Z / A) / E
    prefactor = 78.5 * (density * z_eff / a_eff) / E
    stopping_power = prefactor * np.log(argument)
    
    return np.clip(stopping_power, 0.05, 50.0)


def generate_stopping_power_plot(output_path: str = "reports/figures/stopping_power_curves.png"):
    """
    Строит график кривых торможения электронов в ключевых полупроводниках для диплома.
    """
    print(" Расчет физических кривых торможения (Joy-Luo / Bethe)...")
    
    energies = np.linspace(1.5, 100.0, 500)  # Спектр от 1.5 до 100 кэВ

    # Реальные физические константы: (Плотность, Z_eff, A_eff, Цвет, Стиль линии)
    materials = {
        "Нитрид галлия (GaN)": (6.15, 19.0, 41.8, "green", "-"),
        "Алмаз (Diamond, C)": (3.51, 6.0, 12.01, "blue", "-"),
        "Карбид кремния (4H-SiC)": (3.21, 10.0, 20.05, "red", "-"),
        "Кремний (Si)": (2.33, 14.0, 28.08, "black", "-")
    }

    plt.figure(figsize=(10, 6.5))

    for name, (rho, z, a, color, style) in materials.items():
        sp = joy_luo_stopping_power(energies, rho, z, a)
        plt.plot(energies, sp, label=name, color=color, linestyle=style, linewidth=2.2)

    # Вертикальные маркеры рабочих энергий ключевых изотопов
    plt.axvline(x=5.7, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Тритий H-3 (5.7 кэВ)")
    plt.axvline(x=17.4, color="purple", linestyle="--", linewidth=1.5, alpha=0.8, label="Никель-63 (17.4 кэВ)")
    plt.axvline(x=49.5, color="brown", linestyle="--", linewidth=1.5, alpha=0.8, label="Углерод-14 (49.5 кэВ)")

    plt.xlabel("Энергия электрона $E_\\beta$, кэВ", fontsize=12)
    plt.ylabel("Тормозная способность $dE/dx$, кэВ / мкм", fontsize=12)
    plt.title("Торможение бета-электронов в полупроводниках (модель Joy-Luo / Bethe)", fontsize=13, pad=15)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.yscale("log")
    plt.xscale("log")  # Двойная логарифмическая шкала — стандарт в ядерной физике!
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" График сохранен в: {output_path}")


if __name__ == "__main__":
    generate_stopping_power_plot()