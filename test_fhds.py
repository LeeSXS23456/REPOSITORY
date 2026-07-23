import pandas as pd
import pickle
from rqdatac import *
from helpfunc_basis import *

# import rqdatac
# rqdatac.init(
#     username = "license",
#     password = "ZZ-u7ZWosqrntc3VY3TJzJLPsb-A0o4zehYoiNpDvIBXiwvRIOUmFe7medtMhwu4qiaNxqFSc6ONdGcGeVYgUVd-w5QKScPkmzBEmYVEt94lz9sQZoHwdtQXWWRGGrJqtr7ehiQACydlPS7RcPBfJrpyeTJFsGF1E1guZbpLnvU=XouX9YSi7Pcyo0rSLCMydvHs3nrVq6Rwjda-jI9H_gfGlp53ot0ZnIA6g-ZtvwPDAb62K38pHIqYYyTAyER7FBtZ5HumXzOrWW42LHpUn5-vbnLMxiwbimJ9ns41CaMbjpFEgNcfO52l5wiqDqFCkZNy_OKSDjepfa9GxHsLZZE="
# )
# today = "20260708"
# #更新20260703的基差基本数据
# df_info_m = pd.read_excel("test_fhds.xlsx")
# print(df_info_m.head())

# df_res,df_detail = cal_fhds(today,df_info_m,return_detail=True)
# df_select = df_detail[df_detail["prefix"]=="IC"]
# print(df_select.sort_values("ex_date"))

srcdir = "E:/SJTU/intern/gtht/barra/data_base"
desdir = "E:/SJTU/intern/gtht/基差监控面板"
BASIS_DIR = "E:/SJTU/intern/gtht/barra/data_base/basis/index_future_basis_data.pkl"
def _load_basis():
    if os.path.exists(BASIS_DIR):
        try:
            bd = pd.read_pickle(BASIS_DIR)
            if isinstance(bd, pd.DataFrame) and not bd.empty:
                for _cn in ["date", "listed_date", "maturity_date"]:
                    if _cn in bd.columns:
                        bd[_cn] = pd.to_datetime(bd[_cn])
                return bd
        except:
            pass
    return pd.DataFrame()

CONT = "IF2609"
idx_map = {"IC": "000905", "IM": "000852", "IH": "000016", "IF": "000300"}
_prefix = str(CONT)[:2].upper()
index = idx_map[_prefix] + ".XSHG"

df_basis = _load_basis()
dates = sorted(df_basis.loc[CONT]["date"].unique())

idx_close_df = pd.read_pickle(f"{srcdir}/index/股指宽基指数收盘价_2601_2607.pkl")
with open(f"{srcdir}/index_component_日频/{index}_20_26D_dict.pkl", 'rb') as f:
    weight_dict = pickle.load(f)
with open(f"{srcdir}/index_component_日频/{index}成分股价_26D_dict.pkl", 'rb') as f:
    price_dict = pickle.load(f)

df_dps = pd.read_pickle(f"{srcdir}/basis/866011.RI_dps_22_25H.pkl")
df_eps = pd.read_pickle(f"{srcdir}/basis/866011.RI_eps_24_25Q.pkl")
df_time = pd.read_pickle(f"{srcdir}/basis/dividend_timeline.pkl")


result,detail = cal_fhds_history(CONT,dates,idx_close_df,weight_dict,price_dict,df_dps,df_eps,df_time,return_detail=True)

df_res = df_basis.loc[CONT].merge(result,on="date")
df_res["adj_basis"] = df_res["basis"] + df_res["dividend_point"]
df_res["adj_abs_ratio"] = df_res["adj_basis"] / df_res["close_index"]
df_res["adj_ana_cost"] = df_res["adj_abs_ratio"] / df_res["residual_day"] * 365

df_res.to_excel(f"{desdir}/{CONT}历史分红点数结果.xlsx")
detail.to_excel(f"{desdir}/{CONT}历史分红点数细节.xlsx")

