import numpy as np
import pandas as pd
from pathlib import Path

# ========== 可选画图 ==========
HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MATPLOTLIB = False
    print("未检测到 matplotlib，程序将继续运行，但不会显示图像。")


class SpectrumSimulator:
    """
    单种粒子 / 单一电离态 理论光谱模拟器

    功能：
    1. 从 output/output.xlsx 读取谱线数据
    2. 计算离散相对强度
    3. 支持统一线宽 or Stark 展宽
    4. 构造 Lorentz 峰
    5. 叠加总谱并归一化
    6. 导出结果
    """

    KB_EV = 8.617333262145e-5      # eV/K
    CM1_TO_EV = 1.0 / 8065.54429   # cm^-1 -> eV

    def __init__(self, temperature_k: float):
        self.temperature_k = temperature_k
        self.lines_df = None
        self.discrete_df = None
        self.sim_wavelength = None
        self.sim_intensity = None

    @classmethod
    def cm1_to_ev(cls, energy_cm1):
        return energy_cm1 * cls.CM1_TO_EV

    @staticmethod
    def lorentz_profile(wavelength_grid, center_nm, fwhm_nm):
        """
        面积归一化 Lorentz 线型：
        L(λ) = (1/pi) * [ (Δλ/2) / ((λ-λ0)^2 + (Δλ/2)^2) ]
        """
        if fwhm_nm <= 0:
            raise ValueError(f"fwhm_nm 必须 > 0，当前值为 {fwhm_nm}")

        gamma = fwhm_nm / 2.0
        return (1.0 / np.pi) * (gamma / ((wavelength_grid - center_nm) ** 2 + gamma ** 2))

    def load_lines_from_excel(self, excel_path, sheet_name=0):
        """
        读取 Excel
        必需列：
            wavelength_nm, Aki, Ek_cm1
        优先使用：
            g_upper, E_upper_eV
        若没有，则尝试：
            g_upper = 2*J_k + 1
            E_upper_eV = Ek_cm1 / 8065.54429
        """
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        required_cols = ["wavelength_nm", "Aki", "Ek_cm1"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Excel 缺少必要列: {col}")

        # 数值化
        numeric_cols = ["wavelength_nm", "Aki", "Ek_cm1", "J_k", "g_upper", "E_upper_eV", "stark_w"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 清洗基础列
        df = df.dropna(subset=["wavelength_nm", "Aki", "Ek_cm1"]).copy()
        df = df[(df["wavelength_nm"] > 0) & (df["Aki"] > 0)].copy()

        # 处理 g_upper
        if "g_upper" not in df.columns or df["g_upper"].isna().all():
            if "J_k" not in df.columns:
                raise ValueError("既没有 g_upper，也没有 J_k，无法得到上能级统计权重")
            df["g_upper"] = 2.0 * df["J_k"] + 1.0

        # 处理 E_upper_eV
        if "E_upper_eV" not in df.columns or df["E_upper_eV"].isna().all():
            df["E_upper_eV"] = self.cm1_to_ev(df["Ek_cm1"])

        df = df.dropna(subset=["g_upper", "E_upper_eV"]).copy()
        df = df[(df["g_upper"] > 0)].copy()

        self.lines_df = df.reset_index(drop=True)
        return self.lines_df

    def calculate_discrete_relative_intensities(self, use_wavelength_factor=True):
        """
        第一步：计算离散相对强度

        默认：
            I_j = (A_j * g_j / lambda_j) * exp(-E_j / (k_B T))

        可选关闭：
            I_j = A_j * g_j * exp(-E_j / (k_B T))
        """
        if self.lines_df is None:
            raise ValueError("请先读取 Excel 数据")

        df = self.lines_df.copy()

        boltzmann_factor = np.exp(-df["E_upper_eV"] / (self.KB_EV * self.temperature_k))

        if use_wavelength_factor:
            raw_intensity = (df["Aki"] * df["g_upper"] / df["wavelength_nm"]) * boltzmann_factor
        else:
            raw_intensity = (df["Aki"] * df["g_upper"]) * boltzmann_factor

        df["I_raw_recalc"] = raw_intensity

        max_intensity = df["I_raw_recalc"].max()
        if max_intensity <= 0:
            raise ValueError("重算后的最大强度 <= 0，请检查输入数据或温度")

        df["I_rel_recalc"] = df["I_raw_recalc"] / max_intensity

        self.discrete_df = df
        return self.discrete_df

    def assign_linewidth_constant(self, constant_fwhm_nm=0.05):
        """
        第二步：设定统一线宽
        Δλ_j = Δλ_const
        """
        if self.discrete_df is None:
            raise ValueError("请先计算离散相对强度")

        if constant_fwhm_nm <= 0:
            raise ValueError("constant_fwhm_nm 必须 > 0")

        df = self.discrete_df.copy()
        df["fwhm_nm"] = constant_fwhm_nm
        self.discrete_df = df
        return self.discrete_df

    def assign_linewidth_stark(self, electron_density_cm3=1e16):
        """
        第二步：设定 Stark 线宽
        Δλ_j = 2 * w_j * (n_e / 1e16)

        要求 Excel 中有 stark_w 列
        """
        if self.discrete_df is None:
            raise ValueError("请先计算离散相对强度")

        df = self.discrete_df.copy()

        if "stark_w" not in df.columns:
            raise ValueError("Excel 中没有 stark_w 列，无法使用 Stark 展宽模式")

        if electron_density_cm3 <= 0:
            raise ValueError("electron_density_cm3 必须 > 0")

        df["fwhm_nm"] = 2.0 * df["stark_w"] * (electron_density_cm3 / 1e16)

        # 去掉无效线宽
        df = df.dropna(subset=["fwhm_nm"]).copy()
        df = df[df["fwhm_nm"] > 0].copy()

        self.discrete_df = df.reset_index(drop=True)
        return self.discrete_df

    def build_continuous_spectrum(self, wavelength_min, wavelength_max, step_nm=0.01):
        """
        第三步：构造 Lorentz 峰
        第四步：叠加总谱
        第五步：归一化

        L_j(λ) = (1/pi) * [ (Δλ_j/2) / ((λ-λ_j)^2 + (Δλ_j/2)^2) ]
        S(λ) = Σ I_j * L_j(λ)
        S_norm(λ) = S(λ) / max(S)
        """
        if self.discrete_df is None:
            raise ValueError("请先完成离散强度计算和线宽设定")

        if "fwhm_nm" not in self.discrete_df.columns:
            raise ValueError("请先调用 assign_linewidth_constant() 或 assign_linewidth_stark()")

        if wavelength_max <= wavelength_min:
            raise ValueError("wavelength_max 必须大于 wavelength_min")
        if step_nm <= 0:
            raise ValueError("step_nm 必须 > 0")

        wavelength_grid = np.arange(wavelength_min, wavelength_max + step_nm, step_nm)
        spectrum = np.zeros_like(wavelength_grid, dtype=float)

        for _, row in self.discrete_df.iterrows():
            center = row["wavelength_nm"]
            intensity = row["I_rel_recalc"]
            fwhm_nm = row["fwhm_nm"]

            profile = self.lorentz_profile(wavelength_grid, center, fwhm_nm)
            spectrum += intensity * profile

        max_spec = spectrum.max()
        if max_spec > 0:
            spectrum = spectrum / max_spec

        self.sim_wavelength = wavelength_grid
        self.sim_intensity = spectrum
        return wavelength_grid, spectrum

    def export_discrete_to_excel(self, output_path):
        if self.discrete_df is None:
            raise ValueError("请先生成离散结果")
        self.discrete_df.to_excel(output_path, index=False)

    def export_continuous_to_excel(self, output_path):
        if self.sim_wavelength is None or self.sim_intensity is None:
            raise ValueError("请先生成连续谱")
        out_df = pd.DataFrame({
            "wavelength_nm": self.sim_wavelength,
            "intensity": self.sim_intensity
        })
        out_df.to_excel(output_path, index=False)

    def plot_discrete_lines(self, top_n=None):
        if not HAS_MATPLOTLIB:
            print("未安装 matplotlib，跳过离散谱线绘图。")
            return

        if self.discrete_df is None:
            raise ValueError("请先生成离散结果")

        df = self.discrete_df.copy()

        if top_n is not None:
            df = df.sort_values("I_rel_recalc", ascending=False).head(top_n)
            df = df.sort_values("wavelength_nm")

        plt.figure(figsize=(12, 5))
        plt.vlines(df["wavelength_nm"], 0, df["I_rel_recalc"])
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Relative Intensity")
        plt.title("Discrete Line Spectrum")
        plt.tight_layout()
        plt.show()

    def plot_continuous_spectrum(self):
        if not HAS_MATPLOTLIB:
            print("未安装 matplotlib，跳过连续光谱绘图。")
            return

        if self.sim_wavelength is None or self.sim_intensity is None:
            raise ValueError("请先生成连续谱")

        plt.figure(figsize=(12, 5))
        plt.plot(self.sim_wavelength, self.sim_intensity)
        #plt.xlim(240, 280)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Intensity")
        plt.title("Simulated Continuous Spectrum")
        plt.tight_layout()
        plt.show()


def main():
    # ========= 路径设置 =========
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"

    input_file = output_dir / "output.xlsx"
    discrete_output_file = output_dir / "discrete_lines_output.xlsx"
    continuous_output_file = output_dir / "continuous_spectrum_output.xlsx"

    # ========= 参数设置 =========
    temperature_k = 11600.0
    use_wavelength_factor = True

    # 光谱范围
    wavelength_min = 200.0
    wavelength_max = 900.0
    step_nm = 0.1

    # 线宽模式： "constant" 或 "stark"
    linewidth_mode = "constant"

    # 统一线宽
    constant_fwhm_nm = 0.1

    # Stark 展宽参数
    electron_density_cm3 = 1e16

    # 是否画图
    #plot_top_n_discrete = 50

    # ========= 初始化 =========
    simulator = SpectrumSimulator(temperature_k=temperature_k)

    # ========= 读取数据 =========
    df = simulator.load_lines_from_excel(input_file)
    print("成功读取谱线数：", len(df))
    print(df.head())

    # ========= 第一步：计算离散相对强度 =========
    discrete_df = simulator.calculate_discrete_relative_intensities(
        use_wavelength_factor=use_wavelength_factor
    )

    print("\n离散谱线前 5 行：")
    show_cols = ["wavelength_nm", "Aki", "g_upper", "Ek_cm1", "E_upper_eV", "I_raw_recalc", "I_rel_recalc"]
    print(discrete_df[show_cols].head())

    # ========= 第二步：设定线宽 =========
    if linewidth_mode == "constant":
        simulator.assign_linewidth_constant(constant_fwhm_nm=constant_fwhm_nm)
        print(f"\n当前使用统一线宽模式：FWHM = {constant_fwhm_nm} nm")
    elif linewidth_mode == "stark":
        simulator.assign_linewidth_stark(electron_density_cm3=electron_density_cm3)
        print(f"\n当前使用 Stark 展宽模式：n_e = {electron_density_cm3:.3e} cm^-3")
    else:
        raise ValueError("linewidth_mode 只能是 'constant' 或 'stark'")

    simulator.export_discrete_to_excel(discrete_output_file)
    print(f"离散结果已保存到：{discrete_output_file}")

    # ========= 第三、四、五步：构造峰、叠加、归一化 =========
    simulator.build_continuous_spectrum(
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        step_nm=step_nm
    )

    simulator.export_continuous_to_excel(continuous_output_file)
    print(f"连续光谱已保存到：{continuous_output_file}")

    # ========= 画图 =========
    #simulator.plot_discrete_lines(top_n=plot_top_n_discrete)
    simulator.plot_continuous_spectrum()


if __name__ == "__main__":
    main()
