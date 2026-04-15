import pandas as pd
import numpy as np
import os
from functools import reduce
from scipy import stats
import cvxpy as cp
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

srcdir = "E:/SJTU/实习/国泰海通/barra因子/data_base/"
spedir = "E:/SJTU/实习/国泰海通/barra因子/result/"
navdir = "E:/SJTU/实习/国泰海通/barra因子/intermedia/cvxpy净值曲线/"

print("Start")
files = sorted(os.listdir(f"{srcdir}/base_data/000905"))
num_ori = 100 #初始最优化权重向量（前100个股递减）
alpha_name = 'MACD_HIST'
slope, intercept = 0.0169, -0.0312 #根据alpha因子收益对齐的经验

ret_dict = defaultdict(list)#{}
dual_dict = defaultdict(list) 
error_dt = defaultdict(list)

for i in range(len(files)-1): #1 len(files)+5
    #明确时间域
    alpha_dt = files[i]#"2025-04-10.pkl"
    base_dt = files[i+1]
    print(f"获取{alpha_dt[:10]}天的因子数据并收盘买入，{base_dt[:10]}天卖出")


    #读取数据，最优化
    df_base, df_alpha, df_barra, X_center, variance_frq, variance_rq = [
    pd.read_pickle(path) for path in [
        f"{srcdir}/base_data/whole_mkt/{base_dt}",
        f"{srcdir}/alpha/macd/{alpha_dt}",
        f"{srcdir}/barra_data/whole_mkt/{alpha_dt}",
        f"{srcdir}/barra_data/000905标准化2/{alpha_dt}",
        f"{srcdir}/fac_ret_cov/{alpha_dt}",
        f"{srcdir}/spe_ret_cov/{alpha_dt}",
    ]
]
    df_base['order_book_id'] = df_base['code'].str.replace('.SZ','.XSHE').str.replace('.SH','.XSHG')


    #获得因子信号
    dfs = [df_alpha,df_barra,df_base]
    df_reg = reduce(lambda left, right:left.merge(right,on="order_book_id"),dfs)    #t天的alpha、因子暴露和t+1天的收益
    Rhat_series = (intercept + slope * df_reg[alpha_name])
    Rhat = Rhat_series.fillna(Rhat_series.median()).values
    Num = len(df_reg)


    #获得因子暴露矩阵、收益率协方差矩阵
    orth_order = [x for x in variance_frq.index.tolist()[:11] if x != "comovement"]#["beta","momentum","size","non_linear_size","residual_volatility","liquidity","book_to_price","earnings_yield","growth","leverage"]
    ind_order = list(variance_frq.columns[11:].values)
    X_original = df_reg[orth_order+ind_order].values #风格+行业
    stk_order = df_reg["order_book_id"].tolist()
    X_center = X_center.set_index("order_book_id").loc[stk_order,orth_order].values #风格

    F_cov = variance_frq.loc[orth_order+ind_order, orth_order+ind_order].values
    D_diag = variance_rq.reindex(df_reg["code"]).values.ravel()
    sqrtD = np.sqrt(D_diag)
    #R_cov = X_orth @ F_cov @ X_orth.T


    #设置权重向量的初始值【根据因子信号/等权】
    w_ori = np.zeros_like(Rhat)
    idx = np.argsort(Rhat)[::-1][:num_ori]
    w_ori[idx] = np.exp(-np.arange(num_ori)/20)
    w_ori /= w_ori.sum()


    print(f"开始最优化{base_dt[:10]}组合 | {datetime.now()}")
    #turnover = 0.005
    w = cp.Variable(Num)
    w.value = w_ori
    Xo = X_original.T @ w
    Xp = X_center.T @ w
    lam = cp.Parameter(nonneg=True,value=1)
    x_min_param = cp.Parameter(X_center.shape[1])
    x_max_param = cp.Parameter(X_center.shape[1])
    #penalty = cp.sum_squares(cp.pos(Xp - x_max)) + cp.sum_squares(cp.pos(x_min - Xp))
    #gamma = cp.Parameter(nonneg=True)
    
    objective = cp.Minimize(
            lam * (cp.quad_form(Xo, F_cov) + cp.sum_squares(cp.multiply(sqrtD, w))) - cp.sum(cp.multiply(Rhat, w)) #+ gamma*penalty
        )
    constraints = [
            cp.sum(w) == 1,
            w >= 0,
            #w <= 0.01,
            # #cp.abs(w - w0) <= turnover,
            Xp >= x_min_param,
            Xp <= x_max_param
        ]
    prob = cp.Problem(objective, constraints)
    
    for l_val in [0.01,0.1,0.3,0.5,1]:#[0.001,0.1,1,5,10,100]:
        #gamma.value = l_val
        x_min_param.value = -l_val * np.ones(X_center.shape[1])
        x_max_param.value =  l_val * np.ones(X_center.shape[1])

        try:
            prob.solve(solver=cp.OSQP,max_iter=50000,warm_start=True,verbose=True) #solver=cp.OSQP, #verbose=True #max_iter=10000,
            dual_dict[l_val].append([constraints[i].dual_value for i in range(len(constraints))])
            w_opt = w.value

        except:
            print(f"{base_dt[:10]}最优化失败，等权买入")
            error_dt[l_val].append(base_dt[:10])
            w_opt = np.full(Num, 1/Num)

        #w_opt对应的真实收益
        R_true = df_reg.ret.values
        Rp_realized = w_opt @ R_true
        #ret_dict[base_dt]=Rp_realized
        ret_dict[l_val].append(Rp_realized)

    #更新收益率对齐系数
    x = df_reg[alpha_name]
    y = df_reg['ret']
    q_x_low, q_x_high, q_y_low, q_y_high = x.quantile(0.05), x.quantile(0.95), x.quantile(0.05), x.quantile(0.95)
    # 过滤数据
    df_clean = df_reg[
        (df_reg[alpha_name] >= q_x_low) & (df_reg[alpha_name] <= q_x_high) &
        (df_reg['ret'] >= q_y_low) & (df_reg['ret'] <= q_y_high)
    ]
    x_clean = df_clean[alpha_name]
    y_clean = df_clean['ret']
    # 线性拟合（得到斜率、截距）
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)


#分析结果
from optimize_help import *

# 1️⃣ 检查
check_error(error_dt, [0.01,0.1,0.3,0.5,1])
check_exposure(X_center, w_opt)

# 2️⃣ 解析约束
res_dict = analyze_dual(dual_dict, files, orth_order)

# 3️⃣ 构造因子触线表
df_new = build_factor_df(res_dict, orth_order)

# 4️⃣ 画图
plot_factor_touch(df_new, orth_order, spedir)

# 5️⃣ 净值 & 回测
df_500 = pd.read_excel(f"{srcdir}/000905_SH.xlsx")
df_500.index = df_500["日期"].dt.strftime('%Y-%m-%d')
df_500.rename({"涨跌幅":"000905"}, axis=1, inplace=True)

ret_df, ret_cum = build_nav(ret_dict, files, df_500)
plot_nav(ret_cum,navdir)
metrics = backtest_metrics(ret_df, ret_cum)

save_result(ret_cum, metrics, f"{spedir}/组合优化/minmax_不同std_净值.xlsx")