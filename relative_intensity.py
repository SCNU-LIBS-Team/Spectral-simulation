import pandas as pd
import numpy as np

# =========================
# 1. 参数设置
# =========================
excel_path = r"C:\Users\李翰钰\Desktop\Spectral_simulation\data\nist_lines.xlsx"   # 改成你的文件路径
sheet_name = 0                                               # 第一个工作表
T = 5000                                                     # 激发温度，单位 K

# 玻尔兹曼常数（eV/K）
k_B = 8.617333262e-5

# cm^-1 转 eV 的系数
CM1_TO_EV = 1.239841984e-4

# =========================
# 2. 读取 Excel
# =========================
df = pd.read_excel(excel_path, sheet_name=sheet_name)

print("原始列名：", df.columns.tolist())

# 如果你的列名和这里不完全一样，就按实际改
# 例如图里看起来像：
# obs_wl_air(nm), Aki(s^-1), J_k, Ek(cm-1)

# 为了稳妥，统一重命名
df = df.rename(columns={
    'obs_wl_air(nm)': 'wavelength_nm',
    'Aki (s^-1)': 'Aki',
    'Aki(s^-1)': 'Aki',
    'J_k': 'J_k',
    'Ek(cm-1)': 'Ek_cm1',
    'Ek(cm-1 )': 'Ek_cm1'
})

# =========================
# 3. 数据清洗
# =========================

# 只保留我们需要的列
needed_cols = ['wavelength_nm', 'Aki', 'J_k', 'Ek_cm1']
for col in needed_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要列：{col}，请检查 Excel 表头名字。")

df = df[needed_cols].copy()

# 把各列强制转为数值，无法转换的会变成 NaN
for col in needed_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ===== 关键一步：Aki 为空就跳过 =====
# 同时把波长、J_k、Ek_cm1 为空的行也去掉
df = df.dropna(subset=['Aki', 'wavelength_nm', 'J_k', 'Ek_cm1']).copy()

# 如果 Aki<=0，也没有物理意义，去掉
df = df[df['Aki'] > 0].copy()

# =========================
# 4. 衍生列计算
# =========================

# 上能级统计权重 g_k = 2J_k + 1
df['g_upper'] = 2 * df['J_k'] + 1

# 上能级能量转 eV
df['E_upper_eV'] = df['Ek_cm1'] * CM1_TO_EV

# =========================
# 5. 计算相对强度
# =========================
# 玻尔兹曼分布下：
# I_rel ∝ Aki * g_upper / wavelength_nm * exp(-E_upper / (k_B*T))

df['I_raw'] = (df['Aki'] * df['g_upper'] / df['wavelength_nm']) * np.exp(-df['E_upper_eV'] / (k_B * T))
# 归一化
Imax = df['I_raw'].max()
if Imax > 0:
    df['I_rel'] = df['I_raw'] / Imax
else:
    df['I_rel'] = 0.0

# =========================
# 6. 按波长排序
# =========================
df = df.sort_values(by='wavelength_nm').reset_index(drop=True)

# =========================
# 7. 输出结果
# =========================
print("\n清洗并计算后的前几行：")
print(df.head(20))

# 保存结果
output_path = r"C:\Users\李翰钰\Desktop\Spectral_simulation\output\output.xlsx"  # 改成你想保存的路径
df.to_excel(output_path, index=False)

print(f"\n结果已保存到：{output_path}")
