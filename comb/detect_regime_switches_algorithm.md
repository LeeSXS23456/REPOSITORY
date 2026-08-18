# `detect_regime_switches.py` 算法梳理

本文档按脚本真实执行顺序，总结它如何一步步确定 `start_date`、`confirm_date`、`end_date`，以及各个 window 如何配合验证、`regime_before / regime_after` 如何命名。

---

## 1. 总体流程图

```text
读取 Step1 四张表
  │
  ├─ rank_inertia_daily         → 看整体排序是否失稳/重新稳定
  ├─ factor_cross_section_rank  → 看单因子 rank_delta 是否显著迁移
  ├─ factor_ts_metrics_daily    → 看 20/40/60 日收益质量与跨窗口收敛
  └─ spread_metrics_daily       → 看主线方向是否翻转并持续

        │
        ▼
[阶段A] 识别 start_date
  │
  ├─ A1. 在 20 日窗口上找 rank_instability spike
  │      条件：rank_ratio 或 rank_ma5_ratio ≥ RANK_SPIKE_RATIO
  │
  ├─ A2. 在同一天提取单因子迁移候选
  │      strengthening_factors：rank_delta / rank_delta_roll 明显为正
  │      weakening_factors    ：rank_delta / rank_delta_roll 明显为负
  │
  └─ A3. 用 20 日窗口计算 strengthening_factors 的 lead_quality_score
         作用：判断新力量是否已经开始冒头

        │
        ▼
得到候选 start_signal
(start_date, strengthening_factors, weakening_factors, rank_instability ...)

        │
        ▼
[阶段B] 识别 confirm_date
  │
  ├─ B1. 先在三条 spread 中找 dominant_spread
  │      spread_gv / spread_size / spread_sv
  │
  │      要求：
  │      - start_date 前后先发生变号
  │      - 之后在 20/40 日窗口上被持续确认
  │      - 得分最高且超过阈值
  │
  ├─ B2. 根据 dominant_spread + spread_sign
  │      决定中段要重点跟踪的 target_factors
  │      若 spread 不明确，则退回 strengthening_factors 前两名
  │
  ├─ B3. 从 start_date 往后最多看 20 天
  │      在每一天计算 40/60 日窗口是否向 20 日窗口收敛
  │
  │      收敛分数 = 0.6 × overlap_40 + 0.4 × overlap_60
  │
  └─ B4. 取收敛分数最高的那一天作为 confirm_date
         若未来 20 天都未收敛，则 confirm_date = None

        │
        ▼
得到 middle_signal
(confirm_date, convergence_score, dominant_spread, spread_sign, spread_score ...)

        │
        ▼
[阶段C] 识别 end_date
  │
  ├─ C1. 若没有 confirm_date，则直接失败
  │
  ├─ C2. 从 confirm_date 往后最多看 60 天
  │      逐日检查两个条件是否同时成立：
  │
  │      条件1：rank_instability 已回落到 calm 区间
  │             即 rank_ratio 和 rank_ma5_ratio 都较低
  │
  │      条件2：新 regime 已持续足够久
  │             - 若有 dominant_spread：看该 spread 新方向累计持续天数
  │             - 若无 dominant_spread：看 strengthening_factors 是否持续出现在 leader 里
  │
  └─ C3. 第一个同时满足“稳定 + 持续”的日期，就是 end_date
         若 60 天内都未满足，则退回 confirm_date 作为 end_date

        │
        ▼
[阶段D] 命名 before / during / after
  │
  ├─ regime_after
  │    ├─ 优先：用 dominant_spread + spread_sign 命名
  │    └─ 否则：看 end_date 附近 leaders 与 strengthening_factors 的交集
  │             有交集优先交集；否则退回 leaders 前两名
  │
  ├─ regime_before
  │    ├─ 优先：用同一 dominant_spread 在 start_date 之前的方向命名
  │    └─ 否则：看 start_date 前一日的 leaders 命名
  │
  └─ regime_during
       固定标为 transition_mixed
```

---

## 2. `start_date` 是怎么定的

脚本先只在 **20 日窗口** 上找“切换开始”。

### 2.1 整体排序先失稳
调用 `rank_ins_change()`：
- 先把 `rank_instability` 相对过去 60 日中位数做标准化，得到 `rank_ratio`
- 再判断：
  - `rank_ratio >= RANK_SPIKE_RATIO`
  - 或 `rank_ma5_ratio >= RANK_SPIKE_RATIO`

满足就说明：
- 不是单个因子的小波动，
- 而是整个横截面排序结构开始明显重排。

### 2.2 单因子候选必须同步出现
在同一天调用 `factor_rank_shift_candidates()`：
- `rank_delta > 0` 表示因子变强
- `rank_delta < 0` 表示因子变弱
- 同时还看 `rank_delta_roll`，避免只被单日噪声触发

于是得到：
- `strengthening_factors`
- `weakening_factors`

如果两边都空，这个 spike 就丢掉，不算有效开始点。

### 2.3 20 日先导质量只做辅助说明
调用 `factor_quality_lead_signal()`，只在 **20 日窗口** 上看 strengthening 因子的：
- `cum_return`
- `ts_sharpe`
- `ts_vol`

它不直接决定 `start_date`，只是补充说明：
- 新力量是不是已经开始有收益质量上的冒头。

### 2.4 小结

**`start_date` 本质上是：20 日窗口下，整体排序失稳且单因子迁移明确的那一天。**

---

## 3. `confirm_date` 是怎么定的

`confirm_date` 对应中段确认，重点不是继续看失稳，而是看：

> 20 日窗口上的新信号，能不能带动 40/60 日窗口一起靠拢。

### 3.1 先找哪条 spread 最能解释这次切换
调用 `choose_dominant_spread()`，依次检查：
- `spread_gv`
- `spread_size`
- `spread_sv`

每条 spread 都要过两关：
1. `start_date` 前后先发生 **变号**
2. 变号后在 **20/40 日窗口** 上被后续一段时间持续确认

最后选：
- 得分最高的 spread
- 且得分必须超过阈值

如果没有任何 spread 够强，就认为这次切换暂时不能用主线 spread 定性。

### 3.2 决定中段验证的目标因子
调用 `target_factors_for_middle()`：
- 如果已经识别出 `dominant_spread`，就按 `spread_sign` 取对应一侧因子
- 如果没有 `dominant_spread`，就退回 `strengthening_factors[:2]`

这样中段验证就不再盲看所有因子，而是围绕“这次切换最可能的新主线”来做。

### 3.3 未来 20 天里找“跨窗口收敛”最强的一天
调用 `window_convergence_score()`：
- 从 `start_date` 起往后最多看 **20 天**
- 每一天都取 20 日窗口 top 因子，与 `target_factors` 求交集
- 再看这些因子是否也进入：
  - 40 日窗口 top 因子
  - 60 日窗口 top 因子

得分公式：

```text
score = 0.6 × overlap_40 + 0.4 × overlap_60
```

即：
- 更重视 40 日先跟上
- 60 日只做更慢一步的确认

### 3.4 收敛最强的那一天就是 `confirm_date`
- 若未来 20 天能找到明显收敛的一天，就把该日定为 `confirm_date`
- 若 20 天内都没有有效收敛，则 `confirm_date = None`

### 3.5 小结

**`confirm_date` 本质上是：20 日新主线开始被 40/60 日窗口共同验证的最佳确认点。**

---

## 4. `end_date` 是怎么定的

`end_date` 对应“切换完成”的日期，要求同时满足两件事：

1. **旧的重排已经结束，排序重新稳定**
2. **新的主线已经持续，不是短暂脉冲**

### 4.1 从 `confirm_date` 往后最多看 60 天
调用 `confirm_transition_end()`：
- 不再全样本遍历
- 只在 `confirm_date` 之后最多扫描 **60 天**

这就是结束阶段的搜索边界。

### 4.2 检查“稳定”
调用 `rank_ins_stabilized()`：
- 本质上还是看 `rank_ins_change()` 里的 calm 条件
- 即 `rank_ratio` 和 `rank_ma5_ratio` 都要回落

含义是：
- 开始阶段看“失稳”
- 结束阶段看“再稳定”

### 4.3 检查“持续”
分两种情况：

- **若有 `dominant_spread`**
  - 调用 `new_regime_persistence()`
  - 看从 `start_date` 到当前候选日之间，spread 是否持续站在新方向上

- **若没有 `dominant_spread`**
  - 调用 `factor_regime_persistence()`
  - 看 `strengthening_factors` 是否持续出现在 20 日 leader 中

### 4.4 第一个同时满足“稳定 + 持续”的日期就是 `end_date`
要求：
- `rank_ins_stabilized(date) == True`
- `persistence_days >= SPREAD_PERSISTENCE_DAYS`

一旦满足，当前日期就是 `end_date`。

如果未来 60 天都不满足：
- 脚本会退回 `confirm_date` 作为 `end_date`
- 同时 `stabilized = False`

### 4.5 小结

**`end_date` 本质上是：新主线已经站稳、旧重排已经收敛的第一个完成点。**

---

## 5. 各个 window 是如何配合的

脚本里三个主窗口分工非常明确。

### 5.1 20 日窗口：最前沿，负责“发现”
主要用在：
- 找 `start_date`
- 提取 `strengthening / weakening` 候选
- 判断 spread 是否在短窗口上先翻转
- 判断 leader 是否已经换人

角色：**先导窗口**。

### 5.2 40 日窗口：第一确认层，负责“验证扩散”
主要用在：
- 判断 20 日主线是否向中期扩散
- spread 是否不仅短期翻转，而且中期也支持
- 收敛分数里权重最高（0.6）

角色：**中段主确认窗口**。

### 5.3 60 日窗口：第二确认层，负责“防噪声”
主要用在：
- 检查 60 日是否至少不再与 20 日冲突
- 结束阶段把搜索范围限制在最多 60 天
- 给中段验证再加一层更慢的确认

角色：**慢变量确认窗口**。

### 5.4 一句话概括

```text
20日：先发现
40日：先跟上
60日：再确认
```

也就是：
- 如果只有 20 日在动，更像短扰动
- 当 40/60 日开始向 20 日靠拢，才更像真实的中期 regime 切换

---

## 6. `regime_before` 和 `regime_after` 是怎么定的

### 6.1 `regime_after`
优先级如下：

1. **优先用 spread 命名**
   - 若 `dominant_spread` 明确，就直接用 `dominant_spread + spread_sign` 命名

2. **如果 spread 不明确，再退回因子命名**
   - 先看 `end_date` 附近的 leaders
   - 再与 `strengthening_factors` 求交集
   - **有交集就优先交集**
   - **没交集就退回 leaders 前两名**

所以 `regime_after` 代表的是：
- 最终站稳的新主导风格
- 且尽量和开始阶段识别出的 strengthening 因子保持一致

### 6.2 `regime_before`
优先级如下：

1. **优先用同一条 spread 的旧方向命名**
   - 看 `start_date` 之前，该 spread 的上一方向是什么
   - 然后映射成旧 regime 名称

2. **如果 spread 不明确，再退回因子命名**
   - 直接取 `start_date - 1` 当天的 leaders

所以 `regime_before` 代表的是：
- 切换开始前，旧主导结构对应的风格标签

### 6.3 `regime_during`
脚本当前固定写成：
- `transition_mixed`

含义是：
- 中间阶段更多是过渡态
- 不强行给它贴成一个稳定 regime 名称

---

## 7. 一句话总结脚本主链

```text
先用 20 日 rank_instability 抓“开始失稳”，
再用 20→40→60 的跨窗口收敛抓“中段确认”，
最后用“排序回稳 + 新方向持续”抓“结束完成”，
并优先用 spread 命名前后 regime，不够清晰时再退回单因子/因子组命名。
```
