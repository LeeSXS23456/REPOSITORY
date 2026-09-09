"""
实验脚本：遍历 nsize 计算方法 + 截面回归方法的所有组合，
提取 size & nsize 因子收益的相关系数等关键指标，生成 md 报告。

实验维度：
  nsize_version : size3 / lnmv3          (2)
  include_bjse  : True / False            (2)
  reg_weighted  : True / False            (2)
  weight_type   : sqrt_cap / equal        (2)
共 16 组实验
"""
import os, sys, json, bisect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # 无头模式，不弹窗
import matplotlib.pyplot as plt
from CrossSection import CrossSection

# ===== 路径配置 =====
ROOT = r'E:\SJTU\intern\gtht\barra'
BARRA_BASE  = os.path.join(ROOT, r'通联数据源\因子暴露\格式对齐')
NSIZE_BASE  = os.path.join(ROOT, r'通联数据源\因子暴露\格式对齐_nsize')
RQ_CSV      = os.path.join(ROOT, r'通联数据源\因子暴露\rq_barra_exp_demo.csv')
RET_DIR     = os.path.join(ROOT, r'data_base\stk_ret')
MCP_DIR     = os.path.join(ROOT, r'data_base\stk_mcp_gb')
ZGB_DIR     = os.path.join(ROOT, r'data_base\stk_mcp_zgb')
OUT_DIR     = os.path.join(ROOT, r'通联数据源\nsize实验汇总')
os.makedirs(OUT_DIR, exist_ok=True)

# Step 2/3 抽样区间（nsize 统计 & vs 米筐 只用这个区间，加速）
SAMPLE_START = '2026-08-10'
SAMPLE_END   = '2026-08-14'

# ===== 实验参数 =====
NSIZE_VERSIONS = ['size3', 'lnmv3']
BJSE_OPTIONS   = [True, False]
REG_WEIGHTED   = [True, False]
WEIGHT_TYPES   = ['sqrt_cap', 'equal']

# 收益率区间（含首不含尾的思路：start_date 是第一个收益日）
START_RET = '2026-01-02'
END_RET   = '2026-08-18'

# 因子名（在 factor_ret 输出中 size / non_linear_size 的位置）
SIZE_COL = 'size'
NSIZE_COL = 'non_linear_size'

# ===== 工具函数 =====
def clip_3std(x):
    mu, sigma = x.mean(), x.std()
    return np.clip(x, mu - 3 * sigma, mu + 3 * sigma)

def weighted_stdize(x, w):
    w_mean = np.nansum(x * w) / np.nansum(w)
    std    = np.nanstd(x)
    return np.clip((x - w_mean) / std, -3, 3)

def load_quarters(directory, date_strs):
    quarters = sorted({d[:4] + 'Q' + str((int(d[5:7]) - 1) // 3 + 1) for d in date_strs})
    return pd.concat([pd.read_parquet(f'{directory}/{q}.parquet') for q in quarters])

def bjse_label(b):
    return 'withBJ' if b else 'noBJ'

def w_label(w):
    return 'wls' if w else 'ols'

# ===== Step 1: 计算 nsize 暴露 =====
def calc_nsize(nsize_version, include_bjse, reg_weighted):
    """计算 nsize 因子暴露，若已存在则跳过"""
    sub = f'{nsize_version}_{bjse_label(include_bjse)}_{w_label(reg_weighted)}'
    out = os.path.join(NSIZE_BASE, sub)
    if os.path.isdir(out) and len(os.listdir(out)) > 100:
        print(f'  [skip] {sub} 已存在')
        return out

    os.makedirs(out, exist_ok=True)
    files = sorted(os.listdir(BARRA_BASE))
    dates = [f[:10] for f in files]
    df_zgb = load_quarters(ZGB_DIR, dates)
    df_gb  = load_quarters(MCP_DIR, dates)

    for fname in files:
        dt = fname[:10]
        barra = pd.read_pickle(os.path.join(BARRA_BASE, fname))
        zgb_s = df_zgb.loc[dt].dropna()
        gb_s  = df_gb.loc[dt].dropna()
        common = set(barra['order_book_id']) & set(zgb_s.index) & set(gb_s.index)
        barra = barra[barra['order_book_id'].isin(common)].reset_index(drop=True)
        codes = barra['order_book_id']
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
            valid = zgb > 0
            ln_mv = np.where(valid, np.log(np.where(valid, zgb, np.nan)), np.nan)
            sz = ln_mv.copy()
            nl = ln_mv ** 3
            sz[valid] = clip_3std(sz[valid])
            nl[valid] = clip_3std(nl[valid])
        else:
            sz = barra['size'].values.astype(float)
            nl = sz ** 3
            valid = ~np.isnan(sz)
            m, s = nl[valid].mean(), nl[valid].std()
            nl = np.where(valid, np.clip(nl, m - 3*s, m + 3*s), np.nan)

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
        nsize = weighted_stdize(resid, wt)

        barra['non_linear_size'] = nsize
        barra.to_pickle(os.path.join(out, fname))

    print(f'  [done] {sub} 计算完成')
    return out

# ===== Step 2: nsize 描述统计 =====
def nsize_stats(nsize_dir, include_bjse):
    """抽样区间内，取首末两个交易日算 nsize 描述统计"""
    files = sorted(os.listdir(nsize_dir))
    sample_files = [f for f in files if SAMPLE_START <= f[:10] <= SAMPLE_END]
    stats_rows = []
    for fname in [sample_files[0], sample_files[-1]]:
        df = pd.read_pickle(os.path.join(nsize_dir, fname))
        s = df['non_linear_size'].dropna()
        stats_rows.append({
            'date': fname[:10],
            'n': len(s),
            'mean': s.mean(),
            'std': s.std(),
            'min': s.min(),
            'q25': s.quantile(0.25),
            'median': s.median(),
            'q75': s.quantile(0.75),
            'max': s.max(),
        })
    # 算 size 和 nsize 的截面相关系数（取抽样末日）
    df = pd.read_pickle(os.path.join(nsize_dir, sample_files[-1]))
    corr_sz_ns = df['size'].corr(df['non_linear_size'])
    return stats_rows, corr_sz_ns

# ===== Step 3: vs 米筐 nsize 差异 =====
def vs_rq_nsize(nsize_dir, include_bjse):
    """对比通联自算 nsize vs 米筐 nsize，只用抽样区间"""
    files = sorted(os.listdir(nsize_dir))
    sample_files = [f for f in files if SAMPLE_START <= f[:10] <= SAMPLE_END]
    # 读通联
    rows = []
    for f in sample_files:
        d = pd.read_pickle(os.path.join(nsize_dir, f))
        d['date'] = pd.to_datetime(f[:10])
        d['stock'] = d['order_book_id'].astype(str).str.split('.').str[0].str.zfill(6)
        rows.append(d[['date', 'stock', 'non_linear_size']])
    df_tl = pd.concat(rows, ignore_index=True).rename(columns={'non_linear_size': 'SIZENL_TLY'})

    # 读米筐
    df_rq = pd.read_csv(RQ_CSV)
    df_rq['date'] = pd.to_datetime(df_rq['date'])
    df_rq['stock'] = df_rq['order_book_id'].astype(str).str.split('.').str[0].str.zfill(6)
    df_rq = df_rq.rename(columns={'non_linear_size': 'SIZENL_RQ'})[['date', 'stock', 'SIZENL_RQ']]

    merged = df_tl.merge(df_rq, on=['date', 'stock'], how='inner')
    if len(merged) == 0:
        return {'mean_diff': np.nan, 'corr': np.nan, 'n_dates': 0, 'n_stocks': 0}

    diff = merged['SIZENL_TLY'] - merged['SIZENL_RQ']
    corr = merged['SIZENL_TLY'].corr(merged['SIZENL_RQ'])
    return {
        'mean_diff': diff.mean(),
        'std_diff': diff.std(),
        'corr': corr,
        'n_dates': merged['date'].nunique(),
        'n_stocks': merged['stock'].nunique(),
    }

# ===== Step 4: 计算因子收益，提取 size vs nsize 相关系数 =====
def calc_factor_ret(nsize_dir, weight_type, include_bjse, start_ret, end_ret):
    """跑截面回归，返回 size 与 nsize 的因子收益时序相关系数"""
    files = sorted(os.listdir(nsize_dir))
    dates_list = [f[:10] for f in files]

    # 定位起始日（需要前一天的暴露）
    idx = bisect.bisect_left(dates_list, start_ret)
    if idx == 0:
        raise ValueError(f'start_ret={start_ret} 太早')
    files_sub = files[idx - 1:]
    if end_ret:
        files_sub = [f for f in files_sub if f[:10] <= end_ret]
    ret_dates = [f[:10] for f in files_sub[1:]]

    df_ret = load_quarters(RET_DIR, ret_dates)
    df_mcp = load_quarters(MCP_DIR, ret_dates)

    facret_lst = []
    for i in range(len(files_sub) - 1):
        date_t = files_sub[i + 1][:10]
        barra = pd.read_pickle(os.path.join(nsize_dir, files_sub[i])).dropna(how='any')
        if not include_bjse:
            barra = barra[~barra['order_book_id'].str.endswith('.BJSE')]
        ret_s = df_ret.loc[date_t].dropna()
        mcp_s = df_mcp.loc[date_t].dropna()
        common = set(barra['order_book_id']) & set(ret_s.index) & set(mcp_s.index)
        barra = barra[barra['order_book_id'].isin(common)].reset_index(drop=True)
        codes = barra['order_book_id']
        ret_s = ret_s.loc[codes]
        mcp_s = mcp_s.loc[codes]

        base = pd.DataFrame({
            'code': codes.values,
            'ret': ret_s.values - 0.015 / 252,
            'capital': mcp_s.values,
            'tradadate': date_t,
        })
        cs = CrossSection(base, barra.iloc[:, 13:], barra.iloc[:, 2:12], weight_type=weight_type)
        factor_ret, _, _, _ = cs.reg()
        facret_lst.append(factor_ret)

    df_fac = pd.concat(facret_lst, axis=1).T
    df_fac.index = ret_dates

    # 提取 size 和 nsize 的因子收益
    size_ret = df_fac[SIZE_COL]
    nsize_ret = df_fac[NSIZE_COL]
    corr = size_ret.corr(nsize_ret)

    return {
        'size_ret_mean': size_ret.mean(),
        'nsize_ret_mean': nsize_ret.mean(),
        'size_nsize_corr': corr,
        'n_days': len(ret_dates),
        'cum_size': (1 + size_ret).cumprod().iloc[-1] - 1,
        'cum_nsize': (1 + nsize_ret).cumprod().iloc[-1] - 1,
    }, df_fac

# ===== 主循环：遍历所有组合 =====
all_results = []
total = len(NSIZE_VERSIONS) * len(BJSE_OPTIONS) * len(REG_WEIGHTED) * len(WEIGHT_TYPES)
cnt = 0

for nsize_v in NSIZE_VERSIONS:
    for bjse in BJSE_OPTIONS:
        for reg_w in REG_WEIGHTED:
            tag_nsize = f'{nsize_v}_{bjse_label(bjse)}_{w_label(reg_w)}'
            print(f'\n=== {tag_nsize} ===')

            # Step 1: 算 nsize 暴露
            nsize_dir = calc_nsize(nsize_v, bjse, reg_w)

            # Step 2: nsize 描述统计
            stats_rows, corr_sz_ns = nsize_stats(nsize_dir, bjse)

            # Step 3: vs 米筐
            vs_rq = vs_rq_nsize(nsize_dir, bjse)

            for wtype in WEIGHT_TYPES:
                cnt += 1
                exp_tag = f'{tag_nsize}_{wtype}'
                print(f'  [{cnt}/{total}] {exp_tag} ... ', end='')

                # Step 4: 因子收益 + 相关系数
                fret_stats, df_fac = calc_factor_ret(nsize_dir, wtype, bjse, START_RET, END_RET)

                row = {
                    '实验标签': exp_tag,
                    'nsize_version': nsize_v,
                    'include_bjse': bjse_label(bjse),
                    'reg_weighted': w_label(reg_w),
                    'weight_type': wtype,
                    # nsize 暴露统计（末日）
                    'nsize_mean': stats_rows[-1]['mean'],
                    'nsize_std': stats_rows[-1]['std'],
                    'nsize_min': stats_rows[-1]['min'],
                    'nsize_q25': stats_rows[-1]['q25'],
                    'nsize_median': stats_rows[-1]['median'],
                    'nsize_q75': stats_rows[-1]['q75'],
                    'nsize_max': stats_rows[-1]['max'],
                    'size_nsize_expo_corr': corr_sz_ns,
                    # vs 米筐
                    'vs_rq_mean_diff': vs_rq['mean_diff'],
                    'vs_rq_corr': vs_rq['corr'],
                    # 因子收益
                    'n_days': fret_stats['n_days'],
                    'size_ret_mean': fret_stats['size_ret_mean'],
                    'nsize_ret_mean': fret_stats['nsize_ret_mean'],
                    'cum_size': fret_stats['cum_size'],
                    'cum_nsize': fret_stats['cum_nsize'],
                    'size_nsize_ret_corr': fret_stats['size_nsize_corr'],
                }
                all_results.append(row)

                # 保存关键图：size & nsize 累计净值对比
                fig, ax = plt.subplots(figsize=(10, 5))
                df_cum = (1 + df_fac[[SIZE_COL, NSIZE_COL]]).cumprod()
                df_cum.plot(ax=ax, linewidth=1.2)
                ax.set_title(f'{exp_tag}  size vs nsize 累计净值 (corr={fret_stats["size_nsize_corr"]:.3f})')
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, f'{exp_tag}_size_nsize_cum.png'), dpi=100)
                plt.close(fig)

                print(f'corr={fret_stats["size_nsize_corr"]:.3f}')

# ===== 汇总表 =====
df_res = pd.DataFrame(all_results)
df_res.to_csv(os.path.join(OUT_DIR, '实验汇总表.csv'), index=False, encoding='utf-8-sig', float_format='%.4f')

# ===== 生成 md 报告 =====
md_lines = []
md_lines.append('# nsize 计算方法 & 截面回归方法 对比实验')
md_lines.append('')
md_lines.append(f'- 收益率区间：{START_RET} ~ {END_RET}')
md_lines.append(f'- 实验组数：{total}')
md_lines.append('')

md_lines.append('## 核心指标汇总表')
md_lines.append('')
md_lines.append('按 size & nsize 因子收益相关系数排序：')
md_lines.append('')
cols_show = ['实验标签', 'nsize_version', 'include_bjse', 'reg_weighted', 'weight_type',
             'size_nsize_expo_corr', 'vs_rq_corr', 'vs_rq_mean_diff',
             'size_nsize_ret_corr', 'cum_size', 'cum_nsize']
df_sorted = df_res.sort_values('size_nsize_ret_corr')
md_lines.append(df_sorted[cols_show].to_markdown(index=False, floatfmt='.4f'))
md_lines.append('')

md_lines.append('## 分组讨论')
md_lines.append('')

# 按维度分组看 size_nsize_ret_corr
for dim, label in [('nsize_version', 'nsize 版本'),
                    ('include_bjse', '北交所'),
                    ('reg_weighted', 'nsize回归权重'),
                    ('weight_type', '截面回归权重')]:
    md_lines.append(f'### {label}')
    md_lines.append('')
    g = df_res.groupby(dim)['size_nsize_ret_corr'].agg(['mean', 'min', 'max', 'std']).reset_index()
    md_lines.append(g.to_markdown(index=False, floatfmt='.4f'))
    md_lines.append('')

md_lines.append('## 各组 size vs nsize 累计净值图')
md_lines.append('')
for _, row in df_sorted.iterrows():
    tag = row['实验标签']
    corr = row['size_nsize_ret_corr']
    img = f'{tag}_size_nsize_cum.png'
    md_lines.append(f'### {tag}  (corr={corr:.3f})')
    md_lines.append('')
    md_lines.append(f'![{tag}]({img})')
    md_lines.append('')

with open(os.path.join(OUT_DIR, '实验报告.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print(f'\n全部完成！结果在：{OUT_DIR}')
print(f'汇总表：实验汇总表.csv')
print(f'报告：实验报告.md')
