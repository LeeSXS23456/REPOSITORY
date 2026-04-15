import os 
from CrossSection import *
import matplotlib.pyplot as plt
import numpy as np
srcdir = 'E:/SJTU/实习/国泰海通/barra因子/data_base'
desdir = 'E:/SJTU/实习/国泰海通/barra因子/result'
area = "whole_mkt"

print("Start")
files = sorted(os.listdir(f"{srcdir}/barra_data")) #全量
#files = [f for f in files if f[:10] > "2025-10-31"] #增量数据
facret_lst = []
speret_lst = []
R2_dict = {}
purexp_dict = {}

for file_base, file_barra in zip(files[1:], files[:-1]):
    base_data = pd.read_pickle(f"{srcdir}/base_data/{area}/{file_base}").dropna(how="any")
    # base_data.iloc[:, 0] = (base_data.iloc[:, 0]
    # .str.replace('.SZ', '.XSHE', regex=False)
    # .str.replace('.SH', '.XSHG', regex=False))

    barra_data = pd.read_pickle(f"{srcdir}/barra_data/{file_barra}").dropna(how="any")
    barra_data = barra_data[barra_data.iloc[:, 1].isin(base_data.iloc[:, 0])]
    base_data = base_data[base_data.iloc[:, 0].isin(barra_data.iloc[:, 1])]
    
    cs = CrossSection(base_data,barra_data.iloc[:,13:],barra_data.iloc[:,2:12])
    factor_ret,specific_ret,R2,pure_factor_portfolio_exposure = cs.reg()
    facret_lst.append(factor_ret)
    speret_lst.append(specific_ret)
    R2_dict[file_base[:10]] = (R2)
    purexp_dict[file_base[:10]] = (pure_factor_portfolio_exposure)

#保存数据
df_facret = pd.concat(facret_lst,axis=1).T
df_facret.index = [f[:10] for f in files[1:]]
df_facret.to_excel(f"{desdir}/{area}因子收益率截止至{file_base[:10]}.xlsx")

df_speret = pd.concat(speret_lst,axis=1).T
df_speret.index = [f[:10] for f in files[1:]]
df_speret.to_excel(f"{desdir}/{area}特质收益率截止至{file_base[:10]}.xlsx")

df_R2 = pd.DataFrame(pd.Series(R2_dict),columns=["R2"]).to_excel(f"{desdir}/{area}截面回归R方截止至{file_base[:10]}.xlsx",index_label="日期")

df_facret = pd.concat(facret_lst,axis=1).T
df_facret.index = [f[:10] for f in files[1:]]
df_facret_cum = (1 + df_facret).cumprod()

#绘图
fig, ax = plt.subplots(figsize=(12, 6))
df_plot = df_facret_cum.iloc[:, 32:]
df_plot.plot(ax=ax, linewidth=1.2)
# ✅ 图例
ax.legend(loc='best', fontsize=9, ncol=2)
# ✅ 横轴刻度稀疏（核心）
n = len(df_plot.index)
step = max(n // 10, 1)   # 最多显示10个刻度
ax.set_xticks(np.arange(0, n, step))
ax.set_xticklabels(df_plot.index[::step], rotation=45)
ax.grid(alpha=0.3)
ax.set_title("Cumulative Factor Returns", fontsize=14)
plt.tight_layout()
plt.savefig(f"{desdir}/{area}累计因子收益净值_截止{file_base[:10]}.png")
plt.show()