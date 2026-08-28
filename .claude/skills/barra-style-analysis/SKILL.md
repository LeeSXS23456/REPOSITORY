# Barra 风格分析基础

针对 A 股 Barra 因子的标准化分析工具集。

**核心宗旨**：通过 Barra 因子的**收益、风格波动、特质波动、切换、联动性**五个维度，还原指定区间/事件的**原因、过程和影响**。
特质波动维度看的是**未知风险因子的暴露程度**，不是"Alpha 环境好坏"。

**触发场景**：Barra 风格区间分析、因子回测、波动率分析（风格因子+特质收益）、相关性分析、风格切换事件、量化超额回撤事件、市场事件的 Barra 视角复盘。

---

## 执行原则（先读这条！）

### Step 0：因子语义校准（强制，分析前必做）

**任何 Barra 风格分析开始之前，必须先读取 `comb/Barra因子知识文档.md` 中风格因子的定义部分（1.2.2 节），过一遍 10 个因子的准确含义，校准完语义再开始计算和解读。**

重点校准以下容易误读的因子（已踩过坑的用 ⚠️ 标记）：

| 因子 | 准确含义 | 常见误解 |
|------|---------|---------|
| ⚠️ **Non-linear Size** | 中盘效应（`(ln MV)^3` 对 Size 回归取残差），捕捉的是中盘股的独立收益 | 当成"小盘因子"，以为 size↔nsize 正相关就是"大小盘普涨" |
| ⚠️ **Momentum** | 过去半年到一年的相对强势/弱势，纯趋势维度因子 | 当成"质量类因子"，和 earnings_yield、低波动混为一谈 |
| **Size** | 大盘股 vs 小盘股收益差异（对数市值），是线性的市值维度 | — |
| **Earnings Yield** | 盈利收益导致的差异（EP+ETOP+CETOP 加权），属于价值/质量维度 | — |
| **Residual Volatility** | 剥离市场风险后的波动率高低，不是简单的 risk-on/off | — |
| **Liquidity** | 交易活跃度不同，方向需结合具体场景判断（换手率高 ≠ 一定涨或跌） | — |

**校准规则**：
1. 解读任何一个因子收益方向/波动/相关性时，**先想因子定义，再想经济含义，最后才是"看起来像什么"**
2. 两个因子同时涨/跌，不能直接归为一类，必须先确认它们是不是同一个维度的——动量和盈利收益率在熊市末期可能同向，但它们是不同因子
3. 拿不准的时候，回知识文档查，不要望文生义

> 触发方式：每次看到 Barra 风格分析任务，**第一件事就是打开知识文档校准因子语义**，不是直接写计算脚本。

### 计算与思考分离

**不要边算边想、不要穿插搜索**。严格分两阶段：

```
Phase 1：计算阶段     →  一个 Python 脚本算完所有 Barra 指标
Phase 2：搜索阶段     →  市场环境还原（A股/海外/政策，调用 byted-web-search）
Phase 3：解读阶段     →  数据现象 + 市场环境 → 原因分析 → 过程 → 影响
```

这样做的原因：

- 避免 tool call 来回切换触发 API burst 限流
- 集中思考更有条理
- 计算结果一次到位，减少重复劳动

### 节奏控制（防 burst 限流）

如果遇到 `API Error: System protection triggered by request burst`，按以下方式处理：

1. **本地计算合并**：把所有 pandas 计算合并到**一个** Python 脚本/一次 Bash 调用里，不要拆成多次
2. **搜索合并**：把多个搜索需求合并到一次 `byted-web-search` 调用里，分批间隔搜索
3. **不要并行**：同一时间只跑一个重计算任务（风格切换检测、波动率长窗口计算都是重活）
4. **退避重试**：真的触发了，等 30~60 秒再试，不要立刻重发
5. **先本地后远程**：所有本地数据处理完了，再考虑联网搜索

### 就事论事原则

**每次分析一个独立事件，不要主动引入其他事件做横向对比。**

- 报告中不要出现"与 2024H1 不同"、"比上次更严重"、"类似上一轮"之类的表述
- 所有结论、解读、启示都基于本次事件本身的数据和市场背景
- 如果用户明确要求对比两个事件，才可以做对比分析
- 易错点/方法论沉淀可以跨事件总结，但面向单个事件的输出报告必须聚焦本次事件本身

原因：
1. 避免"一切都像上次"的偷懒思维——每次事件都有独特性
2. 防止引入不准确的对比（时间区间不同、数据口径不同、背景不同）
3. 保持报告的自洽性：不读其他报告也能完整理解本次事件

---

## 0. 基础信息速查

### 数据路径

| 数据                | 路径                                                     |
| ------------------- | -------------------------------------------------------- |
| 因子收益（主缓存）  | `data_base/fac_ret/whole_mkt/factor_returns_10_2608.pkl` |
| 特质收益（spe_ret） | `data_base/spe_ret/v1/`（按季度 parquet 存储）           |
| Step 1 中间表目录   | `comb/outputs/`                                          |
| 风格切换事件表      | `comb/outputs/regime_switch_events.parquet`              |

### 10 个风格因子

```
beta, book_to_price, earnings_yield, growth,
leverage, liquidity, momentum, non_linear_size,
residual_volatility, size
```

识别方式：列名不含中文字符，且排除 `comovement`。

### 三大价差定义

| 价差                           | 公式                                                   | >0 含义             |
| ------------------------------ | ------------------------------------------------------ | ------------------- |
| spread_gv（成长-价值）         | (growth+momentum)/2 − (book_to_price+earnings_yield)/2 | 成长占优            |
| spread_size（线性-非线性市值） | size − non_linear_size                                 | 大市值溢价占优      |
| spread_sv（系统弹性-特质波动） | (beta+liquidity)/2 − residual_volatility               | 高beta/高流动性占优 |

---

## 1. 计算函数索引表

> 🎯 **定向调用，不要找来找去。** 每个计算需求对应一个明确的函数/脚本。

### 1.1 核心计算函数（自己写也可以，10 行以内搞定）

| 计算需求                | 一句话公式                                    | 输出                                               | 所在章节 |
| ----------------------- | --------------------------------------------- | -------------------------------------------------- | -------- |
| 净值序列                | `(1+df).cumprod() / (1+df.iloc[0]).cumprod()` | DataFrame                                          | §2       |
| 风格波动率（年化）      | `df.rolling(10).std() * np.sqrt(252)`         | DataFrame                                          | §2       |
| 风格波动历史分位        | `vol.rolling(250).apply(rank) `               | DataFrame                                          | §2       |
| Pearson / Spearman 相关 | `df.corr('pearson'/'spearman')`               | DataFrame                                          | §2       |
| 滚动相关                | `df[f1].rolling(w).corr(df[f2])`              | Series                                             | §2       |
| 回测指标（9项）         | 见 §2 完整计算                                | DataFrame（按收益降序）                            | §2       |
| 风格波动横截面分析      | 高位天数 / 平均rank / 离散度                  | dict                                               | §2       |
| **特质波动（一站式）**  | **`vol_pipeline(start, end)`**                | **DataFrame: avg_vol / avg_vol_rank / vol_growth** | **§2**   |
| 特质波动-事件前后对比   | `event_vol_analysis(events, summary, w=5)`    | 事件前后各5天均值                                  | §2       |
| 风格切换检测            | `run_detection(start, end)`                   | events_df, diagnostics_df                          | §3.5     |

> 📌 **特质波动**用 `helpfunc_specificr.py` 里的 `vol_pipeline`，一行搞定。
> 默认滚动窗口 10 日，历史分位回溯 242 日（≈1 年）。

### 1.2 风格切换检测（两步走，但一次调用链）

| 步骤   | 脚本                             | 函数/入口                   | 输入                      | 输出                      |
| ------ | -------------------------------- | --------------------------- | ------------------------- | ------------------------- |
| Step 1 | `comb/compute_regime_metrics.py` | `run()`                     | 因子收益 pickle（自动读） | 4 张 parquet 中间表       |
| Step 2 | `comb/detect_regime_switches.py` | `run_detection(start, end)` | start, end + 中间表       | events_df, diagnostics_df |

> ⚠️ `start` 要往前推 **60~120 天**，否则起点信号检测不到。
>
> ⚠️ **输出后过滤重叠事件**：只保留 `is_overlapping == False` 的行，一行一个独立事件。
>
> ```python
> events_in = events_df[(events_df["start_date"] >= start_date) & (~events_df["is_overlapping"])]
> ```

---

---

## 2. 一次性计算脚本模板

> ⚠️ **跑脚本之前，先过 Step 0 因子语义校准**（读 `comb/Barra因子知识文档.md` 1.2.2 节）。边算边脑补因子含义是错误高发区。
>
> 直接复制，改日期，跑一次出所有结果。

```python
# -*- coding: utf-8 -*-
"""
Barra 风格区间分析 - 一次性计算脚本
改 start_date / end_date，运行即可。
"""
import pandas as pd
import numpy as np
import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "comb"))

# ========== 配置 ==========
start_date = pd.Timestamp("YYYY-MM-DD")
end_date   = pd.Timestamp("YYYY-MM-DD")
sd_padded  = start_date - pd.Timedelta(days=400)  # 波动率 rank 用

# ========== 1. 加载数据 ==========
df_full = pd.read_pickle(r"data_base/fac_ret/whole_mkt/factor_returns_10_2608.pkl")
df_full.index = pd.to_datetime(df_full.index)
style_cols = [c for c in df_full.columns
              if not re.search(r"[一-鿿]", str(c))
              and str(c).lower() != "comovement"]

df_padded = df_full[(df_full.index >= sd_padded) & (df_full.index <= end_date)]
df_view   = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]

print(f"区间: {start_date.date()} ~ {end_date.date()}, {len(df_view)} 个交易日")

# ========== 2. 回测指标 ==========
nav = (1 + df_view[style_cols]).cumprod()
nav = nav / nav.iloc[0]
n_days = len(df_view)
yf = n_days / 252

cum_ret = nav.iloc[-1] - 1
ann_ret = nav.iloc[-1] ** (1/yf) - 1
ann_vol = df_view[style_cols].std() * np.sqrt(252)
sharpe  = ann_ret / ann_vol.replace(0, np.nan)

rmax = nav.cummax()
dd   = (nav - rmax) / rmax
max_dd = dd.min()
calmar = ann_ret / (-max_dd).replace(0, np.nan)
win_rate = (df_view[style_cols] > 0).mean()

bt = pd.DataFrame({
    "累计收益%": (cum_ret*100).round(2),
    "年化收益%": (ann_ret*100).round(2),
    "年化波动%": (ann_vol*100).round(2),
    "夏普": sharpe.round(3),
    "最大回撤%": (max_dd*100).round(2),
    "胜率%": (win_rate*100).round(1),
}).sort_values("累计收益%", ascending=False)

print("\n=== 回测指标（按累计收益降序）===")
print(bt.to_string())

# 分阶段收益演变（三等分，标注具体日期）
print("\n=== 分阶段累计收益演变（前/中/后各1/3）===")
n = len(df_view)
t1 = n // 3
t2 = 2 * n // 3
phase_info = [
    ("初期", df_view.iloc[:t1]),
    ("中期", df_view.iloc[t1:t2]),
    ("末期", df_view.iloc[t2:]),
]
for name, sub in phase_info:
    sd, ed = sub.index[0].date(), sub.index[-1].date()
    print(f"  {name}: {sd} ~ {ed} ({len(sub)}天)")

phase_cum = {}
for name, sub in phase_info:
    sub_nav = (1 + sub[style_cols]).cumprod()
    sub_nav = sub_nav / sub_nav.iloc[0]
    phase_cum[name] = (sub_nav.iloc[-1] - 1) * 100
phase_df = pd.DataFrame(phase_cum).round(2)
phase_df = phase_df.sort_values("初期", ascending=False)
print(phase_df.to_string())

# ========== 3. 波动率（10日滚动，年化，250d 分位）==========
vol = df_padded[style_cols].rolling(10).std() * np.sqrt(252)
rank_250 = vol.rolling(250, min_periods=250).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

vol_v  = vol[vol.index >= start_date]
rank_v = rank_250[rank_250.index >= start_date]

print("\n=== 期末波动率水平 ===")
end_vol = pd.DataFrame({
    "年化波动%": (vol_v.iloc[-1]*100).round(2),
    "250d分位": rank_v.iloc[-1].round(3),
}).sort_values("年化波动%", ascending=False)
print(end_vol.to_string())

# 横截面
thr = 0.8
high_mask = rank_v >= thr
high_days = pd.DataFrame({
    "高位天数": high_mask.sum().astype(int),
    "占比%": (high_mask.mean()*100).round(1),
}).sort_values("高位天数", ascending=False)

all_high  = int(high_mask.all(axis=1).sum())
half_high = int((high_mask.sum(axis=1) >= 6).sum())
avg_rank  = rank_v.mean(axis=1)
cross_std = rank_v.std(axis=1)

print(f"\n=== 波动率横截面（阈值={thr*100:.0f}%分位）===")
print(f"全因子同时高位: {all_high} / {len(vol_v)} ({all_high/len(vol_v)*100:.1f}%)")
print(f"6个以上因子高位: {half_high} / {len(vol_v)} ({half_high/len(vol_v)*100:.1f}%)")
print(f"平均波动率分位: 均值={avg_rank.mean():.3f}, 期末={avg_rank.iloc[-1]:.3f}")
print(f"因子间离散度(分位std): 均值={cross_std.mean():.3f}")
print("\n各因子高位天数:")
print(high_days.to_string())

# ========== 4. 相关性 ==========
pearson  = df_view[style_cols].corr('pearson')
spearman = df_view[style_cols].corr('spearman')

def top_pairs(corr, n=10):
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i,j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]

print("\n=== Pearson 相关 Top 10 ===")
for a,b,r in top_pairs(pearson):
    print(f"  {a:22s} vs {b:22s}: {r:+.3f}")

print("\n=== Spearman 秩相关 Top 10 ===")
for a,b,r in top_pairs(spearman):
    print(f"  {a:22s} vs {b:22s}: {r:+.3f}")

# 分阶段关键因子对相关性变化（写报告时：先列数据 → 再给综合解读）
print("\n=== 分阶段关键因子对相关性变化 ===")
# 选有经济意义的基准对，不要随便选
key_pairs = [
    ("size", "non_linear_size"),
    ("growth", "book_to_price"),
    ("growth", "momentum"),
    ("beta", "residual_volatility"),
    ("size", "earnings_yield"),
    ("liquidity", "residual_volatility"),
    ("book_to_price", "leverage"),
]
for pname, sub in phase_info:
    pcorr = sub[style_cols].corr('pearson')
    print(f"\n  {pname} 关键因子对:")
    for a, b in key_pairs:
        print(f"    {a:22s} vs {b:22s}: {pcorr.loc[a,b]:+.3f}")

# ========== 5. 特质收益波动率 ==========
print("\n[4] 特质收益波动率")
print("-" * 70)

from helpfunc_specificr import vol_pipeline
spe_summary = vol_pipeline(str(start_date.date()), str(end_date.date()),
                           vol_window=10, lookback_days=242, auto_update=False)

sv_avg = spe_summary["avg_vol"]
sr_avg = spe_summary["avg_vol_rank"]

print(f"  平均特质波动(均值): {sv_avg.mean()*100:.4f}%")
print(f"  平均特质波动(峰值): {sv_avg.max()*100:.4f}% ({spe_summary.loc[sv_avg.idxmax(),'date'].date()})")
print(f"  平均历史分位(均值): {sr_avg.mean():.3f}")
print(f"  平均历史分位(峰值): {sr_avg.max():.3f} ({spe_summary.loc[sr_avg.idxmax(),'date'].date()})")

for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
    d = (sr_avg >= thr).sum()
    r = d / len(spe_summary) * 100
    print(f"  分位>={thr*100:.0f}%天数: {d}/{len(spe_summary)} ({r:.1f}%)")

# 多峰检测：双阈值平台期（警戒区 >60% + 高位区 >70%）
def find_platforms(series, threshold, min_days=3):
    """找出连续 >= min_days 天满足 > threshold 的区间，返回 [(start_idx, end_idx), ...]"""
    mask = series.values > threshold
    segs, in_seg, s = [], False, 0
    for i, h in enumerate(mask):
        if h and not in_seg: in_seg = True; s = i
        elif not h and in_seg:
            in_seg = False
            if i - s >= min_days: segs.append((s, i-1))
    if in_seg and len(mask) - s >= min_days: segs.append((s, len(mask)-1))
    return segs

for label, thr in [("高位区", 0.7), ("警戒区", 0.6)]:
    segs = find_platforms(sr_avg, thr)
    print(f"\n  {label}平台期(>=3天, 分位>{thr*100:.0f}%): {len(segs)} 段")
    for s, e in segs:
        seg = spe_summary.iloc[s:e+1]
        peak_i = seg["avg_vol_rank"].idxmax()
        print(f"    {seg.iloc[0]['date'].date()} ~ {seg.iloc[-1]['date'].date()}"
              f" ({e-s+1}天), 均={seg['avg_vol_rank'].mean():.3f}"
              f", 峰={seg.loc[peak_i,'avg_vol_rank']:.3f}({seg.loc[peak_i,'date'].date()})")

# ========== 6. 风格切换检测 ==========
print("\n[5] 风格切换事件")
print("-" * 70)

from compute_regime_metrics import run as compute_step1
from detect_regime_switches import run_detection

compute_step1()  # 增量更新中间表
detect_start = start_date - pd.Timedelta(days=120)
events_df, _ = run_detection(start=detect_start, end=end_date)

events_in = events_df[
    (events_df["start_date"] >= start_date) & (~events_df["is_overlapping"])
].copy()
print(f"区间内独立事件数: {len(events_in)} (已去重叠, medium=高置信)")
if not events_in.empty:
    cols = ["event_id","start_date","confirm_date","end_date","direction",
            "regime_before","regime_after","dominant_spread",
            "strengthening_factors","confidence_level"]
    # 兼容列名（部分版本字段名可能不同）
    cols = [c for c in cols if c in events_in.columns]
    print(events_in[cols].to_string(index=False))

print("\n=== 计算完成 ===")
```

---

## 3. 解读框架（思考阶段用）

拿到计算结果后，按以下结构分析。**先写数据事实，再写判断和解释。**

### 3.1 收益维度（回测排名）

看什么：

- 排名前 3 的因子（领涨）和后 3 的因子（领跌）
- 首尾差距有多大（分化程度）
- 夏普和最大回撤：是高收益高波动，还是低波动稳健收益
- 胜率：趋势性强还是震荡为主
- **分阶段收益演变**：将区间三等分（初期/中期/末期），展示每阶段因子排名变化，观察风格迁移路径。**必须标明每阶段的具体起止日期**（如：初期 2024-01-18~2024-03-08）

### 3.2 风格波动维度

看什么：

- 整体波动环境：平均波动率分位高不高，是不是系统性高波动
- 哪些因子波动特别大（主线矛盾）
- 波动率横截面：是普涨普跌（齐涨齐跌）还是结构分化
- 期末波动率水平：是缓和还是加剧

### 3.3 特质波动维度（选股端波动 / Alpha 环境核心观测）

**这是量化超额回撤分析的核心维度之一。**

**底层逻辑**：特质收益（specific return）= 个股收益中不能被 Barra 风格+行业因子解释的部分。它不是"Alpha 残差"，而是 **pure alpha + 未知风险因子（unknown risk factors）收益** 的混合体。

当特质收益波动率高企时，意味着**未知风险因子暴露了出来**——量化策略在样本内检验时没有识别/考虑这种风险，以为是 Alpha；到了样本外，这部分未知风险开始波动，超额就面临回撤。

> 核心：Alpha 之所以叫 Alpha，就是低波高收益的。特质波动上升，本质上不是"Alpha 环境变差了"，而是"之前以为的 Alpha，其实藏着未被识别的风险因子"。

#### 分位阈值校准（A 股历史经验值）

特质波动率的历史分位分布是**右偏的**——大部分时间在中低位，极端高位只占少数。因此不能按 0~100% 的线性区间简单划分"高低"。基于 2010 年以来全市场样本的经验校准：

| 历史分位区间    | 定性等级          | 含义                               |
| --------------- | ----------------- | ---------------------------------- |
| < 0.30          | 低位 / 舒适区     | 选股环境稳定，Alpha 延续性好       |
| 0.30 ~ 0.50     | 中低位            | 正常波动，可接受                   |
| **0.50 ~ 0.60** | **中等偏上**      | 已需警惕，未知风险开始抬头         |
| **0.60 ~ 0.70** | **偏高 / 警戒区** | 选股环境明显承压，超额大概率回撤   |
| 0.70 ~ 0.80     | 高位              | 未知风险显著暴露                   |
| > 0.80          | 极端高位 / 冲击区 | 典型的超额回撤期，往往对应市场冲击 |

**判断口诀：50 警戒、60 高危、80 极端。** 区间平均分位超过 0.60 就已经属于"高波动环境"，不要等到 0.80 才说高。

#### 观测清单

- **整体水平**：区间平均特质波动率分位（用上表校准定性），不是简单说"高/低"
- **激增事件**：波动率日增长 > 5% 的日期，对应未知风险集中暴露的时点
- **高波动平台期（>70%）**：连续 ≥3 天分位 > 70% 的区间（70% 已经是高位），是超额回撤的危险区
- **警戒区平台期（>60%）**：连续 ≥3 天分位 > 60% 的区间，是比高位更宽的承压区——**很多回撤事件不止 >70% 的平台期，>60% 的平台期更长、更能反映持续压力**，必须同时报告
- **多峰检测**：一个区间内可能有**多波**特质波动高峰（如微盘崩盘一波、监管冲击一波），不要默认单峰结构。必须列出所有平台期（>60% 和 >70% 两级阈值），标注每波的起止、强度、持续天数
- **与风格波动的关系**：是共振（两端都动荡）还是背离（只有选股端在承压），分阶段对应
- **分阶段变化**：不能只说"前/中/后"，要识别每一波高峰的时间点和驱动因素

#### 解读要点

- 分位 < 0.50 → 未知风险稳定，样本内 Alpha 在样本外延续性好
- 分位 0.50~0.60 → 未知风险开始抬头，策略超额可能已出现小幅回撤
- 分位 > 0.60 → 选股环境明显承压，超额回撤几乎是确定事件，只是程度问题
- 特质波动激增（连续数日 >5% 日增）→ 未知风险因子突变，是回撤的领先/同步信号
- 特质波动持续 >0.70 的平台期 → 未知风险持续暴露，超额持续承压，需要模型迭代
- 特质波动 + 风格切换共振 → 最危险：风格暴露和选股端两端同时失控
- 多峰结构解读：第一波通常是导火索事件（如流动性冲击），第二波往往是结构性因素（如监管、业绩），持续时间可能更长但强度更低——两波之间的缓和期是"假修复"，不能轻易判断压力解除

### 3.4 联动性维度（因子共动结构）

> **分析立场**：你是一名贯穿多轮牛熊的资深量化投资经理，同时具备全球 QS 前 50 金融学教授的学术训练。你对风格因子的联动结构有三层理解：
>
> 1. **统计层**：哪些因子共动、哪些对立（数据事实）
> 2. **经济层**：为什么它们会共动——背后共享什么风险源（增长预期？流动性？风险偏好？）
> 3. **组合层**：这种联动结构对因子组合构建、风险分散、Alpha 捕获意味着什么

#### 阈值规范（先读这条！）

- **核心分析只看 |r| > 0.5 的强相关对**，低于此阈值的列入"次强相关（0.4<|r|<0.5）"作为补充参考，不要把弱相关也拿来长篇解读，避免噪声干扰
- 分阶段相关性变化同理：只选 |r| 变化幅度大、或穿越 0.5 阈值的因子对做深入分析
- 表格可以多列，但文字解读必须聚焦强相关对

#### 观测框架（三步走）

**第一步：识别共动群组（clusters）**

- 强正相关（|r|>0.5）有几对 → 哪些因子天然绑定，构成"风格簇"
- 关注跨类别的强相关（如 size 与 growth 正相关）→ 揭示当前市场的"风格打包"方式
- 直觉上应该正相关但实际相关性低的因子对 → 说明当前环境下它们的驱动源脱钩了

**第二步：识别主线矛盾（hedge pairs）**

- 强负相关（|r|>0.5）有几对 → 市场定价的核心对立轴是什么
- |r|>0.5 说明是强主线，|r| 在 0.3~0.5 之间是弱主线，|r|<0.3 说明没有清晰主线
- 经典对立对（如 growth vs BP）的相关性强度在历史上处于什么位置 → 判断当前是极端分化还是常态

**第三步：经济机制解读**

不要停留在"A 和 B 正相关"，要回答"**为什么现在它们相关，相关意味着什么**"：

| 因子对                          | 正相关的经济含义                                                                     | 负相关的经济含义                                 |
| ------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------ |
| size ↔ non_linear_size          | 大盘与中盘高度联动，市值维度是统一的定价轴（注意：nsize 是中盘效应，不是小盘）         | 大盘与中盘脱钩，市值结构内部分化                 |
| growth ↔ momentum               | 成长趋势共振，景气投资有效                                                           | 价值修复 + 动量瓦解（或反之）                    |
| beta ↔ residual_volatility      | 系统性弹性与个股波动共振，全面 risk-on/off                                           | 市场只动指数不动个股（或反之）                   |
| size ↔ earnings_yield           | 大盘价值/红利风格 → 防御性行情                                                       | 大盘成长 vs 小盘价值 的分化行情                  |
| liquidity ↔ residual_volatility | 高换手=高波动，典型题材/散户行情                                                     | 流动性与波动脱钩，结构性分化                     |
| leverage ↔ book_to_price        | 高杠杆=高价值，周期价值风格                                                          | 去杠杆 + 成长重估                                |

> ⚠️ **非显而易见的因子语义，必须先校准再解读**（参考 `[[barra-factor-interpretation-checklist]]`）。典型易错：
> - non_linear_size = 中盘效应（`(ln MV)^3` 对 Size 回归取残差），**不是小盘股**
> - size ↔ nsize 正相关 = 大盘与中盘同涨同跌，不等于"大小盘普涨"——微盘/小票可能在另一端暴跌
> - size 因子为正 + nsize 因子为正，但 spread_size（size-nsize）走阔 → 本质是"大中盘相对微盘的防御溢价"，不是普涨行情

#### 分阶段相关性变化

- 必须选 5~7 对**有经济意义**的关键因子对（不是随便选），展示初期/中期/末期三阶段相关性数值
- 每对都要写"变化趋势与含义"，不能只列数字
- 推荐必选的基准对：size↔nsize、growth↔BP、beta↔residual_volatility、BP↔leverage、liquidity↔residual_volatility

#### 综合解读（放最后！）

> 📌 **核心解读必须放在分阶段变化之后**，先列数据再给结论，不能上来就甩观点。

至少覆盖以下三层：

1. **定价结构识别**：当前市场有几条独立定价轴？每条轴上的核心对立是什么？
2. **反常信号**：有没有直觉上应该相关但实际不相关（或方向相反）的因子对？反常信号往往比正常信号更有信息量（如 nsize↔residual_volatility 负相关 → 系统性流动性冲击而非个股波动）
3. **动态演化**：从分阶段变化中提炼出市场结构的迁移路径（如：从结构分化 → 全面共振 → 逐步缓和）
4. **与收益维度交叉验证**：领涨因子之间是高度正相关（虚假分散）还是低相关（多维度 Alpha）？

### 3.5 切换维度

看什么（列出事件即可，不用深入。**只看 is_overlapping=False 的独立事件**）：

- 区间内有几次独立风格切换（已去重叠）——次数越多说明风格越不稳定
- 每次切换的方向：看 `direction` 字段（strengthening / weakening / mixed）
  - **strengthening**：新主线由强势因子推动，是"进攻型切换"
  - **weakening**：旧主线由弱势因子坍塌，是"防守型/瓦解型切换"
  - **mixed**：新旧力量交织，方向不明确
- `strengthening_factors`：新主线的主导因子（比 regime_after 更具体）
- `dominant_spread`：是否有清晰的价差主线（None 说明无主线，是混乱切换）

#### 置信度校准（A 股历史经验）

基于历史回测，A 股中期风格切换的检测置信度有以下经验规律：

| 置信度     | 历史占比           | 含义                                                     |
| ---------- | ------------------ | -------------------------------------------------------- |
| high       | **极罕见**（< 5%） | 几乎从未出现——只有最极端、最清晰的 regime shift 才会命中 |
| **medium** | **主流**（~60%）   | **已经属于高置信事件**——有完整的失稳→验证→确认三阶段信号 |
| low        | 次要（~35%）       | 信号偏弱，可能只是短期扰动，需结合其他维度判断           |

> ⚠️ 不要因为看到 "medium" 就觉得"中等=不高"。在 A 股风格切换的语境下，**medium 就是高置信**，high 几乎是传说级。
> 判断事件重要性时：medium = 可靠信号，low = 参考信号。

#### 必列字段

输出事件表时至少包含以下字段：
`event_id, start_date, confirm_date, end_date, direction, regime_before, regime_after, strengthening_factors, dominant_spread, confidence_level`

### 3.6 综合解读

**总结构：现象 → 原因 → 过程 → 影响**

#### 3.6.1 现象（两部分）

**① Barra 数据现象**（来自计算结果，3~5 句话总结）

- 收益排名：谁领涨谁领跌，首尾差距多大
- 波动特征：风格波动 + 特质波动的水平与结构
- 切换与联动：切换频次、主线是否清晰、因子间相关性

**② 市场环境还原**（此时调用 `byted-web-search` 联网搜索，按以下维度梳理）

> 📌 时间范围：以 `start_date ~ end_date` 为主，可适度向前延伸一个季度（铺垫背景），不要扯太远。
>
> 🔍 **搜索策略**：分 2~3 次搜索覆盖不同维度（如：A股表现 + 海外市场 + 国内政策），不要一次搜太泛的关键词。每次搜索用日期限定范围。
>
> ✍️ **写作原则**：围绕"一条主线 + 两个侧翼"组织内容，不要写成零散的信息堆砌。主线是 Barra 数据已经呈现出的核心风格矛盾（如：大小盘分化 → 搜索就围绕大小盘展开），侧翼是海外和政策背景。

**一条主线（A 股风格核心矛盾）**

从 Barra 数据的收益排名 / 价差 / 切换事件出发，确定当前区间的**核心风格矛盾**是什么（如：大小盘切换、成长 vs 价值、高波动 vs 低波动），然后围绕这条主线组织市场环境信息：

- **指数表现**：只列与主线相关的宽基指数涨跌（如主线是大小盘，就列上证50/沪深300 vs 中证1000/中证2000/微盘股指数），不用列全部
- **行业表现**：领涨/领跌各 3 个，说明它们对应什么风格特征（不要堆行业名字）
- **资金行为**：北向 / 两融 / ETF 资金流，重点看是否支持主线逻辑
- **成交与情绪**：成交量变化、市场情绪指标（只提与主线相关的）

**两个侧翼**

**侧翼一：海外与跨资产**（只写与主线相关的，不相关就一笔带过）

- 美联储 / 美债：利率走势对 A 股风格的传导路径（影响成长估值？影响外资流向？）
- 美元 / 人民币：汇率变动对风格的影响
- 美股 / 全球：是否有外溢效应（如 AI 行情传导）
- 商品：如与主线无关可一句话略过

**侧翼二：国内政策与产业**（只写与主线直接相关的催化）

- 监管政策：直接影响风格的政策（如量化监管、IPO 制度、减持规则 → 影响小盘/微盘）
- 宏观政策：货币政策、财政政策的边际变化
- 产业催化：直接驱动风格的行业政策或景气变化（如 AI 产业政策 → 成长风格）

> 不要写成流水账。每一条信息都要能回答一个问题："它和当前 Barra 呈现的风格矛盾有什么关系？" 与主线无关的信息即使看起来重要，也只需一句话带过或不提。

#### 3.6.2 原因分析（独立小节，深度分析）

**搜索补充**：在进入原因分析前，追加 1~2 次针对性搜索，关键词聚焦"为什么 / 原因 / 机制 / 经验教训 / 如何预防 / 教训"等，确保原因分析不流于表面，可以重点关注雪球等投资者论坛网站。例如：

- 量化超额回撤事件 → 搜「量化超额回撤 原因 机制 教训」
- 风格切换事件 → 搜「XX 风格切换 原因 机制 历史」
- 政策驱动事件 → 搜「XX 政策 影响 机制 传导路径」

**分析方法**：提出 2~3 个候选解释，每个解释用 2~3 句话讲清**机制**，然后直接给判断（主导 / 次要 / 不支持），不要逐条列"支持证据/反对证据"的冗余叙述。

判断依据是三层交叉验证：

1. Barra 因子表现是否支持（收益方向、波动特征、联动结构）
2. 市场环境是否吻合（指数/行业/资金/政策）
3. 经济逻辑是否自洽（为什么会发生、为什么是现在）

**输出结构**（精简、直击要点）：

```text
核心结论：一句话总结根本原因（如：微盘流动性危机 + 量化拥挤踩踏的复合型回撤）

【解释一：XXX】
- 机制：2~3 句话说清传导链条
- 证据：关键数据点 2~3 个（Barra + 市场各一个就够）
- 判断：主导 / 次要 / 不支持

【解释二：XXX】
（同上）

【解释三：XXX】
（同上）
```

> 深度复盘方法论参考 `barra-regime-review` skill（comb 目录）——用于宏观机制、竞争性假设、跨资产验证的更深度分析。

#### 3.6.3 过程

如果区间内有风格切换事件，按时间线描述：

- 关键节点日期、切换方向
- 与市场事件的对应关系（哪个事件触发了哪次切换）
- 演化路径：从什么风格 → 经过什么 → 最终到什么风格

#### 3.6.4 影响与启示

- 对量化策略：超额回撤的来源、模型需要调整的方向
- 对组合配置：风格暴露的取舍、风险控制要点
- 对后续市场：风格趋势是否延续、需要关注的信号

> ⚠️ **硬性要求**：影响与启示 + 文末小结，都**必须提到特质收益率波动的表现**（水平、分位、激增时点、平台期、对超额回撤的贡献度判断）。特质波动是选股端 Alpha 环境的核心观测，是用户最关心的指标之一，不能只在数据现象部分出现一次就不再提起。特质收益波动率的激增/延续高位往往是发生量化超额回撤事件的“警报”。

### 3.7 矛盾检查（必做）

数据结论 vs 搜索信息，发现矛盾必须指出：

| 矛盾类型     | 检查点                                                                                  |
| ------------ | --------------------------------------------------------------------------------------- |
| 因子表现矛盾 | 搜索说"成长领涨"，但 growth 收益为负或排名靠后                                          |
| 主导因子矛盾 | 搜索说"价值回归"，但 value 类因子不占优 / spread_gv 方向相反                            |
| 风格波动矛盾 | 搜索说"情绪稳定"，但风格因子波动率分位普遍高位                                          |
| 特质波动矛盾 | 搜索说"选股环境好/超额容易"，但特质收益波动率处于高位（说明未知风险在暴露）             |
| 切换方向矛盾 | 搜索说"小切大"，但 size / non_linear_size 方向相反                                      |
| 相关性矛盾   | 搜索说"风格分化"，但因子相关性反而上升                                                  |
| 置信度误读   | 看到 "medium" 就以为是"中等=不高"——在 A 股切换检测中 medium 就是高置信，high 几乎不存在 |

> 矛盾了就说矛盾，不要硬圆。先放数据，再说差异，再给可能解释（指数 vs 因子、行业 vs 风格、区间不同等）。

---

## 4. 易错点清单

1. **波动率忘年化**：默认 ×√252
2. **净值忘归 1**：不同因子不能直接比绝对水平
3. **rank 窗口不足**：算波动率 rank 要往前多取 250+ 天
4. **切换检测 start 太近**：至少往前推 60 天
5. **weakening_factors 误用**：只代表起点日旧结构松动，不是全区间走弱
6. **复利 vs 单利**：累计收益用 `cumprod`，不是 `cumsum`
7. **最大回撤是负数**：展示时注意取绝对值或加负号
8. **计算拆太碎**：一个脚本能算完的不要拆 N 次 tool call，容易触发 burst 限流
9. **边算边搜**：先把所有本地数据算完，再统一做搜索验证
10. **特质波动忘更新数据**：`vol_pipeline` 默认 `auto_update=False`，要最新数据需设为 True
11. **特质波动未年化**：`vol_pipeline` 输出的 avg_vol 是 10 日滚动日频标准差（未年化），不要和年化的风格波动比数值
12. **重复表述**：同一结论放最合适的一节，不要各节都讲
13. **空泛表述**：少说"风险偏好"，多说具体因子和数值
14. **小结遗漏特质波动**：文末小结必须提到特质收益率波动的表现和含义，不能只讲风格因子
15. **non_linear_size 语义误读**：nsize = 中盘效应（`(ln MV)^3` 对 Size 回归取残差），**不是小盘股**。size↔nsize 正相关只说明大盘与中盘联动，不能解读为"大小盘普涨"——微盘/小票可能在另一端暴跌。判断时必须结合 spread_size 和市场背景（指数表现、资金流向）
16. **核心解读放错位置**：联动性章节的综合解读必须放在"分阶段相关性变化"之后（先数据后结论），不能先甩观点再列数据支撑。所有数据维度都遵循"事实 → 分析 → 结论"的顺序
17. **切换日误读为风格转向 = 行情转好**：风格切换事件只说明风格结构变了，不代表市场变好或普涨。size 因子走强可能是"大盘防御性上涨、小票继续暴跌"的资金迁移，不是"大小盘同步上行"。分析切换影响必须看切换前后两端谁在涨、谁在跌，以及整体市场情绪背景
18. **事件区间缺前置背景**：分析区间起点（如 1/18）的切换事件，必须向前看 1~2 周的市场环境（开年情绪、前期走势、催化因素），不能只盯着区间内的数据——切换的方向和含义往往要结合起点前的背景才能读懂
19. **特质波动单峰谬误**：一个区间内的特质波动往往是**多峰结构**，不能默认"冲顶后一路回落"的单峰叙事。必须用两级阈值（>60% 警戒区、>70% 高位区）分别检测所有平台期，列出每波的起止、强度、驱动因素
20. **跨事件对比冲动**：分析单个事件时，不要主动引入其他事件做对比（"和 H1 不同"、"比上次严重"等）。每个事件有独特性，就事论事、聚焦本次。只有用户明确要求对比时，才做跨事件分析
21. **因子乱贴标签**：不要把不同维度的因子随意打包成"XX 类因子"（如把 momentum 归为"质量类"）。每个因子属于哪个维度，严格按 Barra 定义来。不确定时直接说因子名字，不要发明分类
22. **望文生义解读因子**：分析每个因子前，先确认定义再解读，不要凭名字脑补（如 non-linear size ≠ 小盘、residual volatility ≠ risk-on）。参见 `comb/Barra因子知识文档.md` 1.2.2 节
23. **因子方向判断搞反**：看到因子收益为负，不要想当然认为"这个因子代表的东西跌了/承压了"。必须走完整推导链：因子暴露怎么定义 → 正收益 = 高暴露股票跑赢 → 负收益 = 低暴露股票跑赢。典型错误：size < 0 误以为"小盘承压"，实际是**大盘跑输、小盘占优**

---

## 5. 附录：信息来源与输出规范

### 5.1 信息来源（附录必加）

**每篇分析报告末尾必须附"信息来源"附录**，列出所有引用的搜索结果链接，防止幻觉。

格式：

```markdown
## 附录：信息来源

### Barra 因子数据

- 因子收益：rqdatac.get_factor_return(universe="whole_market", method="implicit", model="v1")
- 特质收益：rqdatac.get_specific_return(model="v1", industry_mapping="sws_2021")

### 市场信息来源

1. [标题或来源名](URL) — 用于 XX 部分 / 验证了什么
2. [标题或来源名](URL) — 用于 XX 部分 / 验证了什么
   ...
```

**要求**：

- 所有从搜索中引用的事实（指数涨跌、政策事件、经济数据等）都必须能在来源列表中找到出处
- 同一条信息来自多个来源时，列可信度最高的那个即可
- Barra 计算数据不需要列 URL，说明数据来源接口即可

### 5.2 输出文件规范

**输出路径**：`comb/end_input/summary/{start_date}_{duration_days}.md`

- `start_date`：区间起始日，格式 YYYYMMDD
- `duration_days`：区间交易日天数

示例：分析 2024-01-18 ~ 2024-06-25（102 个交易日），文件名：`20240118_102d.md`

**文档结构**（与解读框架对应）：

1. 一、数据现象（Barra 五维度计算结果，含分阶段收益演变）
2. 二、市场环境还原（一条主线 + 两个侧翼，不要信息堆砌）
3. 三、原因分析（2~3 个候选解释，每段精简直击要点）
4. 四、过程（切换事件时间线，direction 字段标明进攻/防守/混合）
5. 五、影响与启示
6. 六、矛盾检查（必做）
7. 附录：信息来源

---

## 6. 关联资源

- **因子方向检查**：`[[barra-factor-interpretation-checklist]]`（分析前必过）
- **深度复盘**：`barra-regime-review` skill（comb 目录）—— 宏观机制、竞争性假设、跨资产验证
- **联网搜索**：`byted-web-search` skill（合并搜索需求，减少调用次数）
- **因子知识文档**：`comb/Barra因子知识文档.md`
