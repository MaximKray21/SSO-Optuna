import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq

# ================= КОНСТАНТЫ =================
MU = 398600.4415
J2 = 0.00108514777667668
R_E = 6378.137
T_CC = 86400.0
OMEGA_DOT = 1.99106e-7

# Предел машинной точности для решателей (около 14 знаков)
EPS = 1e-14 

# ================= МАТЕМАТИКА =================


def sso_equations(vars, T_dr):
    a, i = vars
    if a < R_E:
        return [1e6, 1e6]
    eq1 = OMEGA_DOT - (
        -1.5
        * J2
        * (R_E / a) ** 2
        * np.sqrt(MU / a**3)
        * np.cos(i)
        * (1 - 1.5 * J2 * (R_E / a) ** 2) ** 2
    )
    eq2 = T_dr - (
        2
        * np.pi
        * (a ** (3/2))
        / np.sqrt(MU)
        - (
            (
                3 * np.pi * (R_E ** 2) * J2 /
                (2 * np.sqrt(MU * a))
            ) * 
            (1 + 5 * (np.cos(i) ** 2))
        )
        # из-за упрощения формулы возникла ошибка при расчете
        # * np.sqrt(a**3 / MU)
        # * (1 - 1.5 * J2 * (R_E / a) ** 2 * (1 - 5 * np.cos(i) ** 2))
    )
    return [eq1, eq2]


def calc_swath_width(a, i, gamma_deg):
    gamma_rad = np.radians(gamma_deg)
    sin_term = (a / R_E) * np.sin(gamma_rad)
    if sin_term > 1.0:
        return np.inf
    term_in_brackets = np.arcsin(sin_term) - gamma_rad
    return (2 * R_E * term_in_brackets) / np.sin(i)


def get_max_gap(L_mv, n, N, k):
    orbits_in_k = min(n, int(np.floor(k * n / N)) + 1)
    longitudes = np.array([(j * L_mv) % (2 * np.pi * R_E) for j in range(orbits_in_k)])
    longitudes = np.sort(longitudes)
    longitudes = np.append(longitudes, longitudes[0] + 2 * np.pi * R_E)
    return np.max(np.diff(longitudes))


# ================= РЕНДЕР ТАБЛИЦ =================


def save_table_as_image(df, filename, colors=None):
    """Отрисовка DataFrame как картинки"""
    fig_width = 24 if len(df.columns) > 8 else 14
    fig_height = max(len(df) * 0.25 + 1.5, 2.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300) 
    ax.axis("off")

    table = ax.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7) # Уменьшенный шрифт, чтобы влезли числа
    table.scale(1, 1.5)

    for j in range(len(df.columns)):
        cell = table[0, j]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#4c72b0")

    if colors:
        for row_idx, color in colors.items():
            for col_idx in range(len(df.columns)):
                table[row_idx + 1, col_idx].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close()


# =====================================================


def main():
    parser = argparse.ArgumentParser(description="Оперативность и макс крен.")
    parser.add_argument("--k-days", type=int, default=6, help="Оперативность (крен не больше гамма макс.)")
    parser.add_argument("--gamma-max", type=float, default=25.0, help="максимальный угол крена")
    args = parser.parse_args()

    K_DAYS = args.k_days
    GAMMA_MAX = args.gamma_max

    print(f"Calc orbits. k={K_DAYS}, gamma_max={GAMMA_MAX}. Precision is 12 digits...")

    os.makedirs("output_tables", exist_ok=True)

    data_table1 = []
    data_table2 = []

    for N in range(1, 26):
        for m in range(N):
            row_t1 = {"N": N, "m": m}

            for n_ps in [14, 15]:
                n = n_ps * N + m
                T_dr = T_CC * N / n

                a_guess = (MU * (T_dr / (2 * np.pi)) ** 2) ** (1 / 3)
                i_guess = np.radians(98.0)

                # xtol=EPS (1e-14)
                # алгоритм Ньютона-Рафсона
                sol = fsolve(
                    sso_equations, [a_guess, i_guess], args=(T_dr,), full_output=True, xtol=EPS
                )

                if sol[2] == 1 and sol[0][0] > R_E:
                    a_osc, i_osc = sol[0]
                    H = a_osc - R_E
                    i_deg = np.degrees(i_osc)

                    L_mv = 2 * np.pi * R_E * (N / n)
                    L_s = 2 * np.pi * R_E * (m / n)

                    # запись таблицы 1
                    prefix = "14_" if n_ps == 14 else "15_"
                    row_t1[prefix + "a"] = f"{a_osc:.4f}"
                    row_t1[prefix + "i"] = f"{i_deg:.4f}"
                    row_t1[prefix + "Ls"] = f"{L_s:.4f}"
                    row_t1[prefix + "Lmv"] = f"{L_mv:.4f}"

                    # --- расчет таблицы 2 ---
                    max_gap_k = get_max_gap(L_mv, n, N, K_DAYS)
                    req_gamma = None
                    
                    def gamma_root_func(g):
                        return calc_swath_width(a_osc, i_osc, g) - max_gap_k
                    
                    if gamma_root_func(0.0) >= 0:
                        req_gamma = 0.0
                    elif gamma_root_func(GAMMA_MAX) >= 0:
                        req_gamma = brentq(gamma_root_func, 0.0, GAMMA_MAX, xtol=EPS)

                    # Проверка правила 10 градусов за N суток
                    max_gap_N = get_max_gap(L_mv, n, N, N)
                    condition_10 = calc_swath_width(a_osc, i_osc, 10.0) >= max_gap_N

                    is_valid = (req_gamma is not None) and condition_10

                    if is_valid:
                        data_table2.append(
                            {
                                "N": N,
                                "n_ps": n_ps,
                                "m": m,
                                "H, км": f"{H:.4f}",
                                "Крен (gamma), град.": f"{req_gamma:.4f}",
                                "_H_val": H,
                            }
                        )
                else:
                    prefix = "14_" if n_ps == 14 else "15_"
                    row_t1[prefix + "a"] = row_t1[prefix + "i"] = row_t1[prefix + "Ls"] = row_t1[prefix + "Lmv"] = "---"

            data_table1.append(row_t1)

    # ================= СОХРАНЕНИЕ ТАБЛИЦ =================
    df1 = pd.DataFrame(data_table1)
    df1.columns = [
        "N (сут)", "m",
        "a (14 вит), км", "i (14 вит), град", "L_сут (14), км", "L_мв (14), км",
        "a (15 вит), км", "i (15 вит), град", "L_сут (15), км", "L_мв (15), км",
    ]

    print("Processing 'table1_ballistics.png'...")
    save_table_as_image(df1, "output_tables/table1_ballistics.png")

    df2 = pd.DataFrame(data_table2)

    if not df2.empty:
        best_idx = df2["_H_val"].idxmin()
        row_colors = {best_idx: "#256a25"} # Темно-зеленый

        df2_clean = df2.drop(columns=["_H_val"])
        df2_clean.columns = ["Период (N)", "Витков (n_пс)", "Сдвиг (m)", "Высота орбиты, км", "Потребный крен, град."]

        print("Processing 'table2_valid_coverage.png' (only right orbits)...")
        save_table_as_image(df2_clean, "output_tables/table2_valid_coverage.png", colors=row_colors)
        print("READY!")
    else:
        print("Warn! we do have right orbits (K_DAYS, GAMMA_MAX). table 2 is empty.")

if __name__ == "__main__":
    main()
