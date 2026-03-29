# `config.json` 说明

本项目固定使用 JSON 配置文件，所以说明写在这里，而不是写进 `config.json` 注释。

## 最小可用示例

```json
{
  "temperature_K": 11600.0,
  "ne_cm3": 1.0e17,
  "element_mole_fractions": {
    "Fe": 1.0
  },
  "wavelength_min_nm": 200.0,
  "wavelength_max_nm": 900.0,
  "delta_lambda_nm": 0.02,
  "intensity_mode": "energy",
  "broadening_mode": "fixed",
  "fixed_fwhm_nm": 0.05,
  "targets": ["Fe", "Fe II"],
  "export_discrete_lines_used": true,
  "data_dirs": {
    "lines": "data/Lines_data",
    "levels": "data/Levels_data",
    "ionization_energies": "data/Ionization_Energies_data"
  },
  "output_dir": "output"
}
```

## 字段说明

- `temperature_K`
  - 等离子体温度，单位固定为 K。
  - v1.0 不支持 eV 形式输入温度。

- `ne_cm3`
  - 电子数密度，单位固定为 `cm^-3`。
  - 程序内部会统一换算到 SI 单位用于 Saha 计算，并在日志中记录该换算。

- `element_mole_fractions`
  - 元素总相对丰度。
  - 键必须是标准化学符号，例如 `Fe`、`Mn`、`Cr`。
  - 所有值之和必须等于 1，允许 `1e-6` 误差。

- `wavelength_min_nm` / `wavelength_max_nm`
  - 主输出波长区间，单位 nm。

- `delta_lambda_nm`
  - 连续谱输出网格步长，单位 nm。

- `intensity_mode`
  - 允许值：`"energy"` 或 `"photon"`。
  - 两者只影响离散谱线面积公式，不影响后续展宽与叠加逻辑。

- `broadening_mode`
  - 允许值：`"fixed"` 或 `"stark"`。
  - v1.0 只有 `"fixed"` 可用。
  - 若选择 `"stark"`，程序会明确抛出 `NotImplementedError`，不会自动回退到固定线宽。

- `fixed_fwhm_nm`
  - 仅在 `broadening_mode = "fixed"` 时使用。
  - v1.0 采用统一固定 FWHM 的 Lorentzian 展宽。

- `targets`
  - 可省略或设为空列表。
  - 合法格式：
    - `"Fe"`：输出该元素总贡献。
    - `"Fe I"`、`"Fe II"`：输出指定电离态贡献。
  - 非法格式会直接报错终止。

- `export_discrete_lines_used`
  - `true` 时导出：
    - `discrete_lines_used.xlsx`
    - `discrete_lines_skipped.xlsx`

- `data_dirs`
  - 三个目录都必须提供，且 v1.0 只支持 `xlsx` 文件：
    - `lines`
    - `levels`
    - `ionization_energies`

- `output_dir`
  - 输出目录。
  - 连续谱和离散线报告会写到这里。
  - 日志文件默认为 `output/run_log.txt`。

## 数据命名建议

- lines：`Fe_lines.xlsx`
- levels：`Fe_one_Levels.xlsx`、`Fe_two_Levels.xlsx`、`Fe_three_Levels.xlsx`
- ionization energies：`Fe_energy.xlsx`

levels 文件名中的电离态支持：

- `one` / `two` / `three`
- `I` / `II` / `III`
- `1` / `2` / `3`

## v1.0 未实现但保留接口的功能

以下模型在 v1.0 中都不会被偷偷替代：

- `broadening_mode = "stark"`
- 仪器展宽
- 自吸收修正
- Voigt 线型

如果你在当前版本选择这些接口，程序会明确报 `NotImplementedError` 并写入日志。
