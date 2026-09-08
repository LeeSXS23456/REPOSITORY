import os
import bisect
from sys import prefix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CrossSection import CrossSection

# ===== 配置 =====
# barra_dir = r'E:\SJTU\intern\gtht\barra\data_base\barra_data\whole_mkt'  # 米筐
tag = "size3_withBJ_wls"
barra_dir = f'E:/SJTU/intern/gtht/barra/通联数据源/因子暴露/格式对齐_nsize/{tag}'       # 通联
ret_dir   = r'E:\SJTU\intern\gtht\barra\data_base\stk_ret'
# mcp_dir   = r'E:\SJTU\intern\gtht\barra\data_base\stk_mcp'       # 流通市值
mcp_dir   = r'E:\SJTU\intern\gtht\barra\data_base\stk_mcp_gb'  # 股本市值
desdir    = r'E:\SJTU\intern\gtht\barra\result\自算因子收益率'
rf_daily  = 0.015 / 252   # 年化无风险利率日度化
weight_type = 'sqrt_cap'  # 'sqrt_cap' / 'cap' / 'equal'
barra_src = 'tl'  # 'rq' / 'tl'，用于输出文件名标识
include_bjse = True   # 是否包含北交所股票

# ===== barra 文件列表（t-1 日暴露） =====
files = sorted(os.listdir(barra_dir))

# 日期范围：指定收益的起止日期（barra 暴露自动向前多取一天）
# 设为 None 表示不限制
start_date = '2026-01-02'   # 第一个收益日（对应 2025-12-31 的暴露）
end_date   = None

if start_date:
    # 找到 start_date 前一个交易日的 barra 文件作为起点
    dates = [f[:10] for f in files]
    idx = bisect.bisect_left(dates, start_date)
    if idx == 0:
        raise ValueError(f'start_date={start_date} 太早，没有前一日的 barra 暴露')
    files = files[idx - 1:]   # 包含前一日暴露

if end_date:
    files = [f for f in files if f[:10] <= end_date]

ret_dates = [f[:10] for f in files[1:]]  # 对应 t 日收益

# ===== 预加载收益 & 市值（涉及的所有季度 parquet） =====
def load_quarters(directory, date_strs):
    quarters = sorted({d[:4] + 'Q' + str((int(d[5:7]) - 1) // 3 + 1) for d in date_strs})
    return pd.concat([pd.read_parquet(f'{directory}/{q}.parquet') for q in quarters])

df_ret = load_quarters(ret_dir, ret_dates)
df_mcp = load_quarters(mcp_dir, ret_dates)

# ===== 主循环：逐日截面回归 =====
facret_lst, speret_lst, R2_dict = [], [], {}

for i in range(len(files) - 1):
    date_t = files[i + 1][:10]

    # t-1 日因子暴露
    barra = pd.read_pickle(f'{barra_dir}/{files[i]}').dropna(how='any')
    if not include_bjse:
        barra = barra[~barra['order_book_id'].str.endswith('.BJSE')]
    # t 日收益、市值
    ret_s = df_ret.loc[date_t].dropna()
    mcp_s = df_mcp.loc[date_t].dropna()

    # 股票池交集（以 barra 顺序为准）
    common = set(barra['order_book_id']) & set(ret_s.index) & set(mcp_s.index)
    barra  = barra[barra['order_book_id'].isin(common)].reset_index(drop=True)
    codes  = barra['order_book_id']
    ret_s  = ret_s.loc[codes]
    mcp_s  = mcp_s.loc[codes]

    # 构造 base_data（超额收益 = 收益 - 无风险利率）
    base = pd.DataFrame({
        'code':      codes.values,
        'ret':       ret_s.values - rf_daily,
        'capital':   mcp_s.values,
        'tradadate': date_t,
    })

    # 列 2:12 = 10 个风格因子，列 13: = 行业哑变量（comovement 是国家因子，类内自构）
    cs = CrossSection(base, barra.iloc[:, 13:], barra.iloc[:, 2:12], weight_type=weight_type)
    factor_ret, specific_ret, R2, _ = cs.reg()
    facret_lst.append(factor_ret)
    speret_lst.append(specific_ret)
    R2_dict[date_t] = R2

# ===== 保存 =====
df_facret = pd.concat(facret_lst, axis=1).T
df_facret.index = ret_dates
mcp_tag = os.path.basename(mcp_dir.rstrip('/\\'))   # 市值来源标识：stk_mcp / stk_mcp_gb
bjse_tag = 'withBJ' if include_bjse else 'noBJ'
tagdir = f'{desdir}/{barra_src}/{tag}'
prefix = f"{mcp_tag}_{weight_type}_{bjse_tag}"
os.makedirs(tagdir, exist_ok=True)
df_facret.to_excel(f'{tagdir}/{prefix}_因子收益率截止至{ret_dates[-1]}.xlsx')

df_speret = pd.concat(speret_lst, axis=1).T
df_speret.index = ret_dates
df_speret.to_excel(f'{tagdir}/{prefix}_特质收益率截止至{ret_dates[-1]}.xlsx')

pd.DataFrame.from_dict(R2_dict, orient='index', columns=['R2']).to_excel(
    f'{tagdir}/{prefix}_截面回归R方截止至{ret_dates[-1]}.xlsx', index_label='日期')

# ===== 风格因子累计净值图 =====
df_cum = (1 + df_facret).cumprod()
n_style = barra.iloc[:, 2:12].shape[1]   # 风格因子数（最后 n_style 列）
fig, ax = plt.subplots(figsize=(12, 6))
df_cum.iloc[:, -n_style:].plot(ax=ax, linewidth=1.2)
ax.legend(loc='best', fontsize=9, ncol=2)
n = len(df_cum)
step = max(n // 10, 1)
ax.set_xticks(np.arange(0, n, step))
ax.set_xticklabels(df_cum.index[::step], rotation=45)
ax.grid(alpha=0.3)
ax.set_title(f'Cumulative Factor Returns ({weight_type})', fontsize=14)
plt.tight_layout()
plt.savefig(f'{tagdir}/{prefix}_累计因子收益净值_截止{ret_dates[-1]}.png')
plt.show()
