"""重新计算通联 Barra 暴露中的 non_linear_size (nsize)
- lnMV 用总市值 stk_mcp_zgb
- 加权用股本市值 stk_mcp_gb
- 回归用原始去极值后的数据，残差再做标准化
- 输出到格式对齐_nsize 目录，其他列保持不变"""
import os
import numpy as np
import pandas as pd

# ===== 配置 =====
barra_dir  = r'E:\SJTU\intern\gtht\barra\通联数据源\因子暴露\格式对齐'
zgb_dir    = r'E:\SJTU\intern\gtht\barra\data_base\stk_mcp_zgb'   # 总市值（算lnMV）
gb_dir     = r'E:\SJTU\intern\gtht\barra\data_base\stk_mcp_gb'    # 股本市值（加权用）
out_dir    = r'E:\SJTU\intern\gtht\barra\通联数据源\因子暴露\格式对齐_nsize'

start_date = '2025-12-31'
end_date   = None
# nsize版本：
#   'lnmv3'  = (lnMV)^3 ~ lnMV 回归取残差（Barra标准）
#   'size3'  = size^3 ~ size 回归取残差（直接用标准化后的size三次方）
nsize_version = 'size3'
include_bjse = True   # 是否包含北交所股票
reg_weighted = False    # 回归是否市值加权（False=等权）
bjse_tag = 'withBJ' if include_bjse else 'noBJ'
w_tag = 'wls' if reg_weighted else 'ols'
out_dir    = f'{out_dir}/{nsize_version}_{bjse_tag}_{w_tag}'
os.makedirs(out_dir, exist_ok=True)

# ===== 工具函数 =====
def clip_3std(x: np.ndarray) -> np.ndarray:
    mu, sigma = x.mean(), x.std()
    return np.clip(x, mu - 3 * sigma, mu + 3 * sigma)

def load_quarters(directory, date_strs):
    quarters = sorted({d[:4] + 'Q' + str((int(d[5:7]) - 1) // 3 + 1) for d in date_strs})
    return pd.concat([pd.read_parquet(f'{directory}/{q}.parquet') for q in quarters])

def weighted_stdize(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """加权均值 + 等权标准差标准化 + 3σ去极值"""
    w_mean = np.nansum(x * w) / np.nansum(w)
    std    = np.nanstd(x)
    z = (x - w_mean) / std
    return np.clip(z, -3, 3)

# ===== 文件列表 & 预加载 =====
files = sorted(os.listdir(barra_dir))
if start_date:
    files = [f for f in files if f[:10] >= start_date]
if end_date:
    files = [f for f in files if f[:10] <= end_date]
dates = [f[:10] for f in files]

df_zgb = load_quarters(zgb_dir, dates)
df_gb  = load_quarters(gb_dir, dates)

# ===== 主循环 =====
for fname in files:
    dt = fname[:10]
    barra = pd.read_pickle(f'{barra_dir}/{fname}')

    zgb_s = df_zgb.loc[dt].dropna()
    gb_s  = df_gb.loc[dt].dropna()

    # 股票池对齐
    common = set(barra['order_book_id']) & set(zgb_s.index) & set(gb_s.index)
    barra  = barra[barra['order_book_id'].isin(common)].reset_index(drop=True)
    codes  = barra['order_book_id']

    if not include_bjse:
        mask = ~codes.str.endswith('.BJSE')
        barra = barra[mask].reset_index(drop=True)
        codes = barra['order_book_id']

    zgb = zgb_s.loc[codes].values.astype(float)
    gb  = gb_s.loc[codes].values.astype(float)

    wt = gb.copy()
    wt_valid = gb > 0
    wt[wt_valid] = clip_3std(wt[wt_valid])

    if nsize_version == 'lnmv3':
        # 版本1：(lnMV)^3 ~ lnMV 回归取残差
        valid = zgb > 0
        ln_mv = np.where(valid, np.log(np.where(valid, zgb, np.nan)), np.nan)
        raw_nls = ln_mv ** 3

        sz = ln_mv.copy()
        nl = raw_nls.copy()
        sz[valid] = clip_3std(sz[valid])
        nl[valid] = clip_3std(nl[valid])
    else:
        # 版本2：size^3 ~ size 回归取残差（用通联已有的size因子）
        sz = barra['size'].values.astype(float)
        nl = sz ** 3
        valid = ~np.isnan(sz)
        # size本身已经标准化了，不再重复去极值，但三次方要做3σ去极值
        nl = np.where(valid, np.clip(nl, nl[valid].mean() - 3*nl[valid].std(),
                                            nl[valid].mean() + 3*nl[valid].std()), np.nan)

    # --- 第2步：回归取残差（加权/等权） ---
    ok = ~(np.isnan(sz) | np.isnan(nl) | np.isnan(wt))
    x, y = sz[ok], nl[ok]
    w = wt[ok] if reg_weighted else np.ones(len(x)) / len(x)

    w_sum = w.sum()
    xw_mu = (x * w).sum() / w_sum
    yw_mu = (y * w).sum() / w_sum
    cov_xy = (x * y * w).sum() / w_sum - xw_mu * yw_mu
    var_x  = (x * x * w).sum() / w_sum - xw_mu ** 2
    beta = cov_xy / var_x
    alpha = yw_mu - beta * xw_mu

    resid = nl - (alpha + beta * sz)

    # --- 第3步：残差做加权均值标准化 + 3σ去极值 ---
    nsize = weighted_stdize(resid, wt)

    # 写回
    barra['non_linear_size'] = nsize
    barra.to_pickle(f'{out_dir}/{fname}')
    print(f'\r{dt}  n={len(barra)}  nsize_std={np.nanstd(nsize):.4f}', end='')

print(f'\n完成，输出到 {out_dir}')
