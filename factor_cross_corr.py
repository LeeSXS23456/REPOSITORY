"""
逐日计算指定因子与其他所有风格因子（不含comovement）的截面相关系数，输出Excel。
"""
import os
import glob
import pandas as pd
from joblib import Parallel, delayed

# ====== 配置 ======
DATA_DIR = r"E:\SJTU\intern\gtht\barra\data_base\barra_data\whole_mkt"
OUT_DIR  = r"E:\SJTU\intern\gtht\barra\result\因子截面相关系数"
TARGET_FACTOR = "momentum"  # 指定因子（风格因子，不含comovement）
METHOD = "spearman"         # 相关系数方法: "pearson" 或 "spearman"
STYLE_FACTORS = [
    "size", "non_linear_size", "momentum", "liquidity",
    "book_to_price", "leverage", "growth", "earnings_yield",
    "beta", "residual_volatility",
]
N_JOBS = -1
# ==================


def corr_one_day(pkl_path: str, target: str, others: list, method: str) -> dict:
    """单日截面相关系数"""
    df = pd.read_pickle(pkl_path)
    date = df["date"].iloc[0]
    row = {"date": date}
    target_s = df[target]
    for f in others:
        row[f] = target_s.corr(df[f], method=method)
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert TARGET_FACTOR in STYLE_FACTORS, f"{TARGET_FACTOR} 不是风格因子"
    assert METHOD in ("pearson", "spearman"), "METHOD 只能是 pearson 或 spearman"

    others = [f for f in STYLE_FACTORS if f != TARGET_FACTOR]
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
    method_label = "皮尔逊" if METHOD == "pearson" else "斯皮尔曼"
    print(f"共 {len(files)} 个交易日，目标因子: {TARGET_FACTOR}，方法: {method_label}")

    rows = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(corr_one_day)(f, TARGET_FACTOR, others, METHOD) for f in files
    )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, f"{TARGET_FACTOR}_截面相关系数_{method_label}.xlsx")
    out.to_excel(out_path, index=False)
    print(f"已输出: {out_path}")
    print(out.head())
    print("...")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()
