import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =======================
# 1️⃣ 基础检查
# =======================
def check_error(error_dt, l_vals):
    print([len(error_dt[l]) for l in l_vals])


def check_exposure(X_center, w_opt):
    print(X_center.T @ w_opt)


# =======================
# 2️⃣ 解析 dual 约束
# =======================
def analyze_dual(dual_dict, files, orth_order):
    res_dict = []
    num_lst = []

    for l_val, day_list in dual_dict.items():
        for dt, daily_vals in zip(files[1:], day_list):
            val0, val1, val2, val3 = daily_vals

            num_lst.append(len(np.where(val1 == 0)[0]))

            # 下限
            val2 = val2.astype(float)
            if val2.max() != 0:
                idx = np.where(val2 > 0)[0]
                factors = [orth_order[i] for i in idx]
                res_dict.append([l_val, dt[:10], "lower_bond", factors])
                print(f"{dt}天的{factors}风格因子【下限】紧约束")

            # 上限
            val3 = val3.astype(float)
            if val3.max() != 0:
                idx = np.where(val3 > 0)[0]
                factors = [orth_order[i] for i in idx]
                res_dict.append([l_val, dt[:10], "upper_bond", factors])
                print(f"{dt}天的{factors}风格因子【上限】紧约束")

    print(f"策略平均每日配股 {np.mean(num_lst), np.median(num_lst)}")

    return res_dict


# =======================
# 3️⃣ one-hot展开
# =======================
def build_factor_df(res_dict, orth_order):
    df = pd.DataFrame(res_dict, columns=["boundary", "tradadate", "type", "factors"])

    res = []
    for _, row in df.iterrows():
        factors = row['factors']
        onehot = [1 if f in factors else 0 for f in orth_order]
        res.append(list(row) + onehot)

    df_new = pd.DataFrame(res, columns=list(df.columns) + orth_order)
    df_new["count"] = df_new[orth_order].sum(axis=1)

    return df_new


# =======================
# 4️⃣ 画图工具
# =======================
def parse_tag(tag):
    b, t = eval(tag)
    return float(b), t


def get_color(boundary, typ, b_min, b_max):
    norm = (boundary - b_min) / (b_max - b_min + 1e-8)
    if typ == 'lower_bond':
        return plt.cm.Blues(0.3 + 0.5 * norm)
    else:
        return plt.cm.Reds(0.3 + 0.7 * norm)


def plot_factor_touch(df_new, orth_order, spedir):
    for fac in orth_order:
        temp = df_new[["boundary", "tradadate", "type", fac]]

        res = []
        for g, df_s in temp.groupby(["boundary", "type"]):
            tag = str(g)
            df_s = df_s.set_index("tradadate").rename({fac: tag}, axis=1)
            res.append(df_s[[tag]])

        df_g = pd.concat(res, axis=1).fillna(0)
        df_g.index = pd.to_datetime(df_g.index)

        boundaries = [parse_tag(c)[0] for c in df_g.columns]
        b_min, b_max = min(boundaries), max(boundaries)

        plt.figure(figsize=(12, 6))

        # scatter
        for col in df_g.columns:
            b, t = parse_tag(col)
            color = get_color(b, t, b_min, b_max)
            plt.scatter(df_g.index, df_g[col], s=10, alpha=0.3, color=color)

        # trend
        df_mean = df_g.expanding().mean()
        for col in df_mean.columns:
            b, t = parse_tag(col)
            color = get_color(b, t, b_min, b_max)
            plt.plot(df_mean.index, df_mean[col], '--', color=color, linewidth=2)

        plt.title(f"{fac} touch")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{spedir}/组合优化/{fac}触线图.png")


# =======================
# 5️⃣ 净值 + 回测
# =======================
def build_nav(ret_dict, files, df_500):
    ret_df = pd.DataFrame(ret_dict, index=[f[:10] for f in files[1:]])
    ret_df = ret_df.merge(df_500[["000905"]], left_index=True, right_index=True)

    temp = pd.DataFrame(0, index=["2025-01-01"], columns=ret_df.columns)
    ret0_df = pd.concat([temp, ret_df])

    ret_cum = (1 + ret0_df).cumprod()
    return ret_df, ret_cum

def plot_nav(ret_cum,path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))

    # 逐列画（避免label问题）
    for col in ret_cum.columns:
        plt.plot(ret_cum.index, ret_cum[col], label=col)

    # x轴刻度控制
    step = max(1, len(ret_cum) // 10)
    plt.xticks(
        ticks=ret_cum.index[::step],
        labels=ret_cum.index[::step],
        rotation=45,
        fontsize=10
    )

    plt.grid(alpha=0.3)
    plt.legend(loc='best')
    plt.title("Portfolio NAV")
    plt.tight_layout()
    plt.savefig(f"{path}/minmax_各个std_原始因子暴露及协方差.png")
    plt.show()
    
def backtest_metrics(ret_df, ret_cum, rf=0.015):
    res = []

    for i in range(ret_cum.shape[1]):
        port_nav = ret_cum.iloc[:, i]

        cum_ret = port_nav.iloc[-1] / port_nav.iloc[0] - 1
        ann_ret = (1 + cum_ret)**(252 / len(port_nav)) - 1
        ann_vol = ret_df.iloc[:, i].std() * np.sqrt(252)

        sp = (ann_ret - rf) / ann_vol
        maxd = (port_nav / port_nav.cummax()).min() - 1
        km = (ann_ret - rf) / abs(maxd)

        res.append(pd.Series({
            '累计收益率': f'{cum_ret:.2%}',
            '年化收益率': f'{ann_ret:.2%}',
            '年化波动率': f'{ann_vol:.2%}',
            '夏普比率': f'{sp:.2f}',
            '最大回撤': f'{maxd:.2%}',
            '卡玛比率': f'{km:.2f}'
        }, name=ret_cum.columns[i]))

    return pd.concat(res, axis=1)


def save_result(ret_cum, metrics, path):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        ret_cum.to_excel(writer, sheet_name='净值')
        metrics.to_excel(writer, sheet_name='回测结果')

    print("Excel 保存成功！✅")