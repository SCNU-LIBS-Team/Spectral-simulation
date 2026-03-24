import re
from pathlib import Path

import numpy as np
import pandas as pd

# ========== 可选画图 ==========
HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MATPLOTLIB = False
    print("未检测到 matplotlib，程序仍可运行，但不会显示或保存图像。")


# =========================================
# 常量设置
# =========================================
KB_EV = 8.617333262145e-5       # 玻尔兹曼常数，eV/K
CM1_TO_EV = 1.239841984e-4      # cm^-1 -> eV
C2 = 1.438776877                # hc/kB，适用于 cm^-1 与 K
k_B_J = 1.380649e-23            # 玻尔兹曼常数，J/K
h = 6.62607015e-34              # 普朗克常数，J*s
m_e = 9.1093837015e-31          # 电子质量，kg

CHI_1 = 7.9024                  # Fe I  -> Fe II 电离能，eV
CHI_2 = 16.1878                 # Fe II -> Fe III 电离能，eV


# =========================================
# 工具函数
# =========================================
def parse_j(value):
    """
    解析 J 值。
    支持：
    4
    2.5
    9/2
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(" ", "")
    if text == "":
        return np.nan

    match = re.fullmatch(r"(\d+)/(\d+)", text)
    if match:
        return float(match.group(1)) / float(match.group(2))

    try:
        return float(text)
    except Exception:
        return np.nan


def parse_level(value):
    """
    解析能级数值。
    会去掉空格和 '?' 后再转为浮点数。
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(" ", "").replace("?", "")
    if text == "":
        return np.nan

    try:
        return float(text)
    except Exception:
        return np.nan


def find_levels_header_row(raw_df):
    """
    自动寻找能级表的真实表头行。
    表头中通常会包含：
    Configuration / J / Level
    """
    for index in range(len(raw_df)):
        row = [str(v).strip() for v in raw_df.iloc[index].tolist()]
        if "Configuration" in row and "J" in row and "Level" in row:
            return index

    raise ValueError("未找到能级表表头，请检查 Excel 文件格式。")


def read_levels_excel(file_path):
    """
    读取单个能级 Excel。
    最终只保留：
    J
    Level_cm1
    """
    raw = pd.read_excel(file_path, header=None)
    header_row = find_levels_header_row(raw)
    df = pd.read_excel(file_path, header=header_row)
    df.columns = [str(col).strip() for col in df.columns]

    j_col = None
    level_col = None

    for column in df.columns:
        column_lower = column.lower()
        if column_lower == "j":
            j_col = column
        elif "level" in column_lower:
            level_col = column

    if j_col is None or level_col is None:
        raise ValueError(f"{file_path} 中未找到 J 或 Level 列。")

    out = df[[j_col, level_col]].copy()
    out.columns = ["J", "Level_cm1"]
    out["J"] = out["J"].apply(parse_j)
    out["Level_cm1"] = out["Level_cm1"].apply(parse_level)

    out = out.dropna(subset=["J", "Level_cm1"]).copy()
    out = out.drop_duplicates(subset=["J", "Level_cm1"]).reset_index(drop=True)
    return out


def calc_partition_function(levels_df, temperature_k):
    """
    计算配分函数：
    U(T) = sum[(2J + 1) * exp(-C2 * E / T)]
    其中 E 的单位为 cm^-1，T 的单位为 K。
    """
    # 物理公式：
    # U(T) = Σ[g_i * exp(-E_i / (k_B T))]
    #
    # 在这里：
    # g_i = 2J_i + 1，为第 i 个能级的统计权重
    # E_i 用 cm^-1 表示，因此把 E_i/(k_B T) 改写成 C2 * E_i / T
    # C2 = hc/k_B，数值上可直接用于 cm^-1 与 K 的组合
    #
    # 物理含义：
    # 配分函数描述某一电离态下所有能级在热平衡时的总布居“分母”。
    # 后续用玻尔兹曼分布求某个上能级布居时，要除以这个 U(T)。
    #
    # 近似条件：
    # 1. 默认满足局域热平衡（LTE）
    # 2. 能级表足够完整；若高能级缺失，U(T) 会偏小
    # 3. 没有额外考虑外场、非平衡激发等效应
    g = 2.0 * levels_df["J"].to_numpy() + 1.0
    energy_cm1 = levels_df["Level_cm1"].to_numpy()
    return float(np.sum(g * np.exp(-C2 * energy_cm1 / temperature_k)))


def find_nist_lines_header_row(raw_df, search_limit=80):
    """
    自动寻找 NIST 谱线表的多行表头起始行。
    一般会包含：
    Ion / Observed
    """
    for index in range(min(len(raw_df), search_limit)):
        row = [str(v).strip() for v in raw_df.iloc[index].tolist()]
        if "Ion" in row and any("Observed" in cell for cell in row):
            return index

    raise ValueError("未找到 NIST 谱线表表头，请检查 Excel 文件格式。")


def read_nist_lines_excel(file_path):
    """
    读取 NIST 谱线 Excel，并提取模拟需要的关键字段。

    当前默认按你这份 NIST 导出格式读取：
    0   -> Ion
    1   -> 观测波长
    6   -> Aki
    10  -> 上能级
    16  -> J_k
    """
    raw = pd.read_excel(file_path, header=None, sheet_name=0)
    header_row = find_nist_lines_header_row(raw)

    # 当前 NIST 导出里，真正数据通常从 header_row + 4 开始
    data = raw.iloc[header_row + 4:].copy().reset_index(drop=True)

    mapping = {
        0: "Ion",
        1: "obs_wl_air_nm",
        6: "Aki",
        10: "Ek_cm1",
        16: "J_k",
    }

    for column_index in mapping:
        if column_index >= data.shape[1]:
            raise ValueError("NIST 谱线表列数不足，当前列映射失效，请重新检查导出格式。")

    df = data[list(mapping.keys())].rename(columns=mapping)

    # 统一电离态写法
    df["Ion"] = (
        df["Ion"]
        .astype(str)
        .str.strip()
        .replace({"FeI": "Fe I", "FeII": "Fe II", "FeIII": "Fe III"})
    )

    # 数值化
    df["obs_wl_air_nm"] = pd.to_numeric(df["obs_wl_air_nm"], errors="coerce")
    df["Aki"] = pd.to_numeric(df["Aki"], errors="coerce")
    df["Ek_cm1"] = df["Ek_cm1"].apply(parse_level)
    df["J_k"] = df["J_k"].apply(parse_j)

    # 清洗，只保留有效的 Fe I / Fe II / Fe III 谱线
    df = df.dropna(subset=["Ion", "obs_wl_air_nm", "Aki", "Ek_cm1", "J_k"]).copy()
    df = df[df["Ion"].isin(["Fe I", "Fe II", "Fe III"])].copy()
    df = df[df["obs_wl_air_nm"] > 0].copy()
    df = df[df["Aki"] > 0].copy()

    # 派生列
    df["ion_stage"] = df["Ion"]
    df["wavelength_nm"] = df["obs_wl_air_nm"]
    df["g_upper"] = 2.0 * df["J_k"] + 1.0
    df["E_upper_eV"] = df["Ek_cm1"] * CM1_TO_EV

    return df.reset_index(drop=True)


def saha_ratio(U_z, U_zm1, chi_eV, temperature_k, electron_density_cm3, delta_Ei_eV=0.0):
    """
    计算 Saha 比值：
    n_z / n_(z-1)
    """
    # 物理公式（Saha 方程）：
    # n_z / n_(z-1)
    # = (2 * U_z / U_zm1)
    #   * (2π m_e k_B T / h^2)^(3/2)
    #   * (1 / n_e)
    #   * exp(-(χ - ΔEi) / (k_B T))
    #
    # 在这里：
    # n_(z-1) 表示低一级电离态粒子数
    # n_z     表示高一级电离态粒子数
    # U_zm1   表示低一级电离态配分函数
    # U_z     表示高一级电离态配分函数
    # χ       为电离能
    # ΔEi     为可选的电离能降低修正
    #
    # 物理含义：
    # 这个比值决定在给定温度和电子密度下，
    # Fe I / Fe II / Fe III 之间应该如何分配相对粒子数。
    #
    # 近似条件：
    # 1. 默认等离子体接近 LTE
    # 2. 默认电离平衡已建立
    # 3. 电子密度和温度在计算区域内视为均匀
    # 4. 未考虑时间演化、复合动力学、输运过程等非平衡效应
    electron_density_m3 = electron_density_cm3 * 1e6
    factor = (
        (2.0 * U_z / U_zm1)
        * ((2.0 * np.pi * m_e * k_B_J * temperature_k) / (h ** 2)) ** 1.5
    )
    exponent = -(chi_eV - delta_Ei_eV) / (KB_EV * temperature_k)
    return factor / electron_density_m3 * np.exp(exponent)


def lorentz_profile(wavelength_grid, center_nm, fwhm_nm):
    """
    面积归一化 Lorentz 线型：
    L(λ) = (1/pi) * [ (Δλ/2) / ((λ-λ0)^2 + (Δλ/2)^2) ]
    """
    # 物理公式：
    # L(λ) = (1/π) * [γ / ((λ - λ0)^2 + γ^2)]
    # 其中 γ = FWHM / 2
    #
    # 物理含义：
    # 把一条原本理想的离散谱线，展宽成具有有限半高宽的连续峰形。
    # 这种线型常用来近似碰撞展宽、压力展宽、Stark 展宽等情形。
    #
    # 近似条件：
    # 1. 默认线型可由 Lorentz 形式近似
    # 2. 当前代码对所有谱线使用统一 FWHM，不区分具体跃迁
    # 3. 未考虑仪器函数、高斯多普勒展宽、Voigt 线型等更复杂情况
    if fwhm_nm <= 0:
        raise ValueError(f"fwhm_nm 必须 > 0，当前值为 {fwhm_nm}")

    gamma = fwhm_nm / 2.0
    return (1.0 / np.pi) * (gamma / ((wavelength_grid - center_nm) ** 2 + gamma ** 2))


def build_continuous_spectrum(lines_df, wavelength_min, wavelength_max, step_nm, fwhm_nm,
                              intensity_column="I_raw_total", normalize=False):
    """
    将所有离散谱线做固定 Lorentz 展宽并叠加，
    得到连续光谱后再归一化。
    """
    if wavelength_max <= wavelength_min:
        raise ValueError("wavelength_max 必须大于 wavelength_min")
    if step_nm <= 0:
        raise ValueError("step_nm 必须 > 0")

    # 物理公式：
    # S(λ) = Σ [ I_j * L_j(λ) ]
    #
    # 其中：
    # I_j    为第 j 条离散谱线强度
    # L_j(λ) 为第 j 条谱线的展宽线型函数
    #
    # 物理含义：
    # 把所有离散谱线的峰形叠加起来，得到连续光谱。
    # 如果 normalize=True，则最后再做：
    # S_norm(λ) = S(λ) / max(S)
    #
    # 近似条件：
    # 1. 各条谱线之间线性叠加
    # 2. 默认忽略自吸收、辐射传输和谱线混合的复杂反馈
    # 3. 使用统一线宽参数时，是数值近似，不代表每条线真实展宽都相同
    wavelength_grid = np.arange(wavelength_min, wavelength_max + step_nm, step_nm)
    spectrum = np.zeros_like(wavelength_grid, dtype=float)

    for _, row in lines_df.iterrows():
        spectrum += row[intensity_column] * lorentz_profile(
            wavelength_grid,
            row["wavelength_nm"],
            fwhm_nm,
        )

    max_spec = spectrum.max()
    if normalize and max_spec > 0:
        spectrum = spectrum / max_spec

    return wavelength_grid, spectrum


def safe_to_excel(df, output_path):
    """
    安全写入 Excel。
    如果文件正在被占用，给出更清楚的提示。
    """
    try:
        df.to_excel(output_path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入文件：{output_path}。该文件可能正在被 Excel 或其他程序占用。"
        ) from exc


def save_and_plot_spectrum(wavelength_grid, intensity, output_png=None, show_plot=False):
    """
    保存并可选显示连续光谱。
    """
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(wavelength_grid, intensity)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.title("Fe I + Fe II + Fe III Simulated Spectrum")
    plt.tight_layout()

    if output_png is not None:
        plt.savefig(output_png, dpi=200)

    if show_plot:
        plt.show()

    plt.close()


def save_stage_plot(wavelength_grid, intensity, title, ylabel, output_png=None, show_plot=False):
    """
    保存单个电离态或总谱图像。
    """
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(12, 5))
    plt.plot(wavelength_grid, intensity)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if output_png is not None:
        plt.savefig(output_png, dpi=200)

    if show_plot:
        plt.show()

    plt.close()


def run_simulation(
    base_dir,
    output_dir,
    temperature_k=11600.0,
    electron_density_cm3=1e17,
    constant_fwhm_nm=0.08,
    wavelength_min=200.0,
    wavelength_max=900.0,
    step_nm=0.02,
    show_plot=False,
    delta_Ei_1=0.0,
    delta_Ei_2=0.0,
):
    """
    主模拟函数：
    1. 读取 Fe I / Fe II / Fe III 能级表
    2. 计算配分函数
    3. 计算 Saha 电离平衡
    4. 读取 NIST 谱线并计算相对强度
    5. 做固定线宽展宽，生成连续光谱
    6. 导出结果并可选画图
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========= 输入文件 =========
    fe1_levels_file = base_dir / "Fe_one_Levels.xlsx"
    fe2_levels_file = base_dir / "Fe_two_Levels.xlsx"
    fe3_levels_file = base_dir / "Fe_three_Levels.xlsx"
    lines_file = base_dir / "nist_lines.xlsx"

    # ========= 输出文件 =========
    discrete_output_file = output_dir / "fe123_discrete_lines.xlsx"
    continuous_output_file = output_dir / "fe123_continuous_spectrum.xlsx"
    plot_output_file = output_dir / "fe123_fixed_width_spectrum.png"
    stage_output_dir = output_dir / "fe123_by_stage"
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    # ========= 第一步：读取能级表并计算配分函数 =========
    fe1_levels = read_levels_excel(fe1_levels_file)
    fe2_levels = read_levels_excel(fe2_levels_file)
    fe3_levels = read_levels_excel(fe3_levels_file)

    U_FeI = calc_partition_function(fe1_levels, temperature_k)
    U_FeII = calc_partition_function(fe2_levels, temperature_k)
    U_FeIII = calc_partition_function(fe3_levels, temperature_k)

    # ========= 第二步：Saha 方程计算三种电离态占比 =========
    R1 = saha_ratio(U_FeII, U_FeI, CHI_1, temperature_k, electron_density_cm3, delta_Ei_1)
    R2 = saha_ratio(U_FeIII, U_FeII, CHI_2, temperature_k, electron_density_cm3, delta_Ei_2)

    n_total = 1.0
    n_FeI = n_total / (1.0 + R1 + R1 * R2)
    n_FeII = R1 * n_FeI
    n_FeIII = R1 * R2 * n_FeI

    U_dict = {"Fe I": U_FeI, "Fe II": U_FeII, "Fe III": U_FeIII}
    n_dict = {"Fe I": n_FeI, "Fe II": n_FeII, "Fe III": n_FeIII}

    # ========= 第三步：读取谱线并计算相对强度 =========
    lines_df = read_nist_lines_excel(lines_file)
    lines_df["U_stage"] = lines_df["ion_stage"].map(U_dict)
    lines_df["n_stage"] = lines_df["ion_stage"].map(n_dict)

    # 相对发射强度模型：
    # I ~ (n_stage / U_stage) * (Aki * g_upper / wavelength) * exp(-E_upper / (k_B*T))
    #
    # 更接近物理本源的写法是：
    # I_ul ∝ n_u * A_ul * hν
    #
    # 又因为在 LTE 条件下：
    # n_u = n_stage * (g_u / U(T)) * exp(-E_u / (k_B T))
    #
    # 并且 ν = c / λ，所以可写成：
    # I_ul ∝ (n_stage / U(T)) * g_u * A_ul * exp(-E_u / (k_B T)) * (1 / λ)
    #
    # 这正是当前代码使用的结构来源。
    #
    # 物理含义：
    # 1. n_stage / U_stage 决定该电离态整体布居背景
    # 2. g_upper 反映上能级统计权重
    # 3. Aki 反映该跃迁自发辐射概率
    # 4. exp(-E_upper / kT) 反映高能级在热平衡下布居随能量升高而减小
    # 5. 1 / λ 对应光子频率因子 ν = c / λ
    #
    # 近似条件：
    # 1. 默认满足 LTE，因此能级布居可用玻尔兹曼分布
    # 2. 默认不同电离态之间满足 Saha 平衡
    # 3. 这里得到的是“理论未标定强度”或 arb. unit，不是实验绝对光强
    # 4. 未显式乘入真实发射体积、几何收集效率、仪器响应、自吸收修正
    boltzmann_factor = np.exp(-lines_df["E_upper_eV"] / (KB_EV * temperature_k))
    lines_df["I_raw_total"] = (
        (lines_df["n_stage"] / lines_df["U_stage"])
        * (lines_df["Aki"] * lines_df["g_upper"] / lines_df["wavelength_nm"])
        * boltzmann_factor
    )

    max_intensity = lines_df["I_raw_total"].max()
    if max_intensity > 0:
        # 物理含义：
        # 这里只是为了绘图和比较方便，将当前数据集中最强谱线缩放到 1。
        # 因此 I_rel_total 是“相对强度”，不是绝对物理单位。
        lines_df["I_rel_total"] = lines_df["I_raw_total"] / max_intensity
    else:
        lines_df["I_rel_total"] = 0.0

    lines_df = lines_df.sort_values(["wavelength_nm", "ion_stage"]).reset_index(drop=True)
    safe_to_excel(lines_df, discrete_output_file)

    # ========= 第四步：总谱固定线宽展宽并生成连续光谱 =========
    wavelength_grid, intensity_raw = build_continuous_spectrum(
        lines_df=lines_df,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        step_nm=step_nm,
        fwhm_nm=constant_fwhm_nm,
        intensity_column="I_raw_total",
        normalize=False,
    )
    _, intensity_rel = build_continuous_spectrum(
        lines_df=lines_df,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        step_nm=step_nm,
        fwhm_nm=constant_fwhm_nm,
        intensity_column="I_raw_total",
        normalize=True,
    )

    continuous_df = pd.DataFrame({
        "wavelength_nm": wavelength_grid,
        "intensity_raw_sum": intensity_raw,
        "intensity_rel_sum": intensity_rel,
    })
    safe_to_excel(continuous_df, continuous_output_file)

    # ========= 第五步：分别生成 Fe I / Fe II / Fe III / Sum 数据和图像 =========
    stage_summaries = {}
    for stage_name in ["Fe I", "Fe II", "Fe III"]:
        stage_key = stage_name.replace(" ", "").lower()
        stage_lines = lines_df[lines_df["ion_stage"] == stage_name].copy()

        if len(stage_lines) == 0:
            continue

        stage_max = stage_lines["I_raw_total"].max()
        if stage_max > 0:
            # 对单个电离态内部再单独归一化，便于观察该电离态内部的强弱分布。
            # 注意：I_rel_stage 只能比较同一电离态内部的相对强弱，
            # 不能直接和其它电离态的 I_rel_stage 做绝对量级比较。
            stage_lines["I_rel_stage"] = stage_lines["I_raw_total"] / stage_max
        else:
            stage_lines["I_rel_stage"] = 0.0

        stage_discrete_file = stage_output_dir / f"{stage_key}_discrete_lines.xlsx"
        safe_to_excel(stage_lines, stage_discrete_file)

        stage_grid, stage_raw = build_continuous_spectrum(
            lines_df=stage_lines,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            step_nm=step_nm,
            fwhm_nm=constant_fwhm_nm,
            intensity_column="I_raw_total",
            normalize=False,
        )
        _, stage_rel = build_continuous_spectrum(
            lines_df=stage_lines,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            step_nm=step_nm,
            fwhm_nm=constant_fwhm_nm,
            intensity_column="I_raw_total",
            normalize=True,
        )

        stage_continuous_df = pd.DataFrame({
            "wavelength_nm": stage_grid,
            "intensity_raw": stage_raw,
            "intensity_rel": stage_rel,
        })
        stage_continuous_file = stage_output_dir / f"{stage_key}_continuous_spectrum.xlsx"
        safe_to_excel(stage_continuous_df, stage_continuous_file)

        stage_plot_file = stage_output_dir / f"{stage_key}_spectrum.png"
        save_stage_plot(
            wavelength_grid=stage_grid,
            intensity=stage_raw,
            title=f"{stage_name} Simulated Spectrum",
            ylabel="Line Intensity (arb. unit)",
            output_png=stage_plot_file,
            show_plot=show_plot,
        )

        stage_summaries[stage_name] = {
            "line_count": int(len(stage_lines)),
            "discrete_output_file": str(stage_discrete_file),
            "continuous_output_file": str(stage_continuous_file),
            "plot_output_file": str(stage_plot_file),
        }

    sum_plot_file = stage_output_dir / "sum_spectrum.png"
    sum_continuous_file = stage_output_dir / "sum_continuous_spectrum.xlsx"
    safe_to_excel(continuous_df, sum_continuous_file)
    save_stage_plot(
        wavelength_grid=wavelength_grid,
        intensity=intensity_raw,
        title="Fe I + Fe II + Fe III Sum Spectrum",
        ylabel="Line Intensity (arb. unit)",
        output_png=sum_plot_file,
        show_plot=show_plot,
    )

    # 保留原有总谱输出文件，兼容之前的使用方式
    save_and_plot_spectrum(
        wavelength_grid=wavelength_grid,
        intensity=intensity_rel,
        output_png=plot_output_file,
        show_plot=False,
    )

    summary = {
        "temperature_k": float(temperature_k),
        "electron_density_cm3": float(electron_density_cm3),
        "fixed_fwhm_nm": float(constant_fwhm_nm),
        "U_FeI": float(U_FeI),
        "U_FeII": float(U_FeII),
        "U_FeIII": float(U_FeIII),
        "n_FeI": float(n_FeI),
        "n_FeII": float(n_FeII),
        "n_FeIII": float(n_FeIII),
        "line_count": int(len(lines_df)),
        "ion_counts": lines_df["ion_stage"].value_counts().to_dict(),
        "discrete_output_file": str(discrete_output_file),
        "continuous_output_file": str(continuous_output_file),
        "plot_output_file": str(plot_output_file),
        "stage_output_dir": str(stage_output_dir),
        "stage_outputs": stage_summaries,
        "sum_continuous_output_file": str(sum_continuous_file),
        "sum_plot_output_file": str(sum_plot_file),
    }
    return summary


def main():
    # ========= 路径设置 =========
    project_dir = Path(__file__).resolve().parent
    base_dir = project_dir / "data"
    output_dir = project_dir / "output"

    # ========= 参数设置 =========
    temperature_k = 11600.0
    electron_density_cm3 = 1e17
    constant_fwhm_nm = 0.3

    wavelength_min = 200.0
    wavelength_max = 900.0
    step_nm = 0.1

    show_plot = False

    # 可选的电离能降低修正
    delta_Ei_1 = 0.0
    delta_Ei_2 = 0.0

    # ========= 开始计算 =========
    summary = run_simulation(
        base_dir=base_dir,
        output_dir=output_dir,
        temperature_k=temperature_k,
        electron_density_cm3=electron_density_cm3,
        constant_fwhm_nm=constant_fwhm_nm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        step_nm=step_nm,
        show_plot=show_plot,
        delta_Ei_1=delta_Ei_1,
        delta_Ei_2=delta_Ei_2,
    )

    # ========= 输出结果 =========
    print("计算完成，结果如下：")
    for key, value in summary.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()
