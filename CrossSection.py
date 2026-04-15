import numpy as np
import pandas as pd
import re
from statsmodels.stats.weightstats import DescrStatsW
testdir = "E:/SJTU/实习/国泰海通/barra因子/intermedia"
print("Version 8")
"""
NOTICE:
rq因子暴露已经标准化，用不用standardiza函数区别不大
√处理异方差现象
√处理多重共线性问题
√因子暴露要滞后一期！

"""

def standardize(exposure:pd.DataFrame,capital:pd.Series):
    """截取，
    利用市值加权（only均值）"""
    median = exposure.median()
    mad = (exposure.sub(median)).abs().median()
    upper = median + 3 * mad
    lower = median - 3 * mad
    exposure_clipped = exposure.clip(lower=lower, upper=upper, axis=1)

    w_obj = DescrStatsW(exposure_clipped,weights=capital)
    w_mu = w_obj.mean
    w_std = exposure.std()
    return (exposure - w_mu) / w_std



class CrossSection():
    """
    base_data:record shares'basics
    industry_expo:0/1
    style_expo:float
    """

    def __init__(self,base_data:pd.DataFrame,industry_expo=pd.DataFrame(),style_expo=pd.DataFrame()):
        self.N = base_data.shape[0]
        self.P = industry_expo.shape[1]
        self.Q = style_expo.shape[1]
        self.date = base_data.tradadate[0]
        self.capital = base_data.capital
        self.ret = base_data.ret
        self.country_tag = ["comovement"]
        self.industry_tag = industry_expo.columns
        self.barra_tag = style_expo.columns
        self.stkids = base_data.code
        
        self.style_expo = style_expo.values#standardize(style_expo,self.capital).values #米筐已经标准化了
        self.industry_expo = industry_expo.values
        self.W = np.diag(np.sqrt(self.capital) / sum(np.sqrt(self.capital))) # already inversed / np.eye(self.N)
        self.country_expo = np.array(self.N*[[1]])

        print('\rCross Section Regression, ' + 'Date: ' + self.date  + ', ' + \
              str(self.N) + ' Stocks, ' + str(self.P) + ' Industry Facotrs, ' +  str(self.Q) + ' Style Facotrs', end = '')

    def check_tags(self):
        # 是否包含英文
        has_english = lambda s: bool(re.search(r'[A-Za-z]', s))
        # 是否包含中文
        has_chinese = lambda s: bool(re.search(r'[\u4e00-\u9fff]', s))

        industry_has_en = any(has_english(x) for x in self.industry_tag)
        barra_has_cn = any(has_chinese(x) for x in self.barra_tag)

        if industry_has_en or barra_has_cn:
            raise ValueError(
                f"标签错误：industry_tag 中不能含英文，barra_tag 中不能含中文\n"
                f"industry_tag={self.industry_tag}\n"
                f"barra_tag={self.barra_tag}"
            )

    def reg(self):
        """横截面回归R=XF+e
        控制异方差和多重共线性问题"""
        industry_cap = self.capital @ self.industry_expo
        X = np.matrix(np.hstack([self.country_expo,self.industry_expo,self.style_expo])) # N*(1+P+Q)

        C = np.eye((1+self.P+self.Q))
        C[self.P,1:(1+self.P)] = -industry_cap / industry_cap[-1]
        C = np.delete(C,self.P,axis=1) # (P+Q+1)*(P+Q)
        #pd.DataFrame(C).to_excel(f"{testdir}/C.xlsx")
        X_trans = X @ C # N*(P+Q)
        pure_factor_weight = C @ np.linalg.inv(X_trans.T @ self.W @ X_trans) @ X_trans.T @ self.W

        pure_factor_ret = pd.Series(np.array(pure_factor_weight @ self.ret).ravel(),index=self.country_tag+list(self.industry_tag) + list(self.barra_tag))
        specific_ret = self.ret - np.array(X @ pure_factor_ret.T).ravel()
        specific_ret.index =self.stkids
        R2 = 1 - np.var(specific_ret) / np.var(self.ret)
        pure_factor_expo = pure_factor_weight @ X # shoule equal I

        return ((pure_factor_ret,specific_ret,R2,pure_factor_expo))