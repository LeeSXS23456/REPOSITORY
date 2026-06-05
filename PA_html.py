import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
srcdir = os.path.join(BASE_DIR, "data_base", "excess_exposure")
df_info = pd.read_excel(os.path.join(srcdir, "实习生净值脱敏产品代码匹配表.xlsx"), index_col=0)

# 添加当前目录到 Python 路径，以便导入 weight_contribution 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="业绩归因分析面板",
    page_icon="📊",
    layout="wide"
)

# 初始化 session_state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'interval_df' not in st.session_state:
    st.session_state.interval_df = None
if 'detailed_df' not in st.session_state:
    st.session_state.detailed_df = None

st.title("📊 业绩归因分析面板")
st.markdown("---")

# 1. 侧边栏参数设置
st.sidebar.header("⚙️ 参数设置")

# 基准指数选择
bmk_options = {
    "沪深300": 300,
    "中证500": 905,
    "中证1000": 852,
    "国证2000": 932000
}
selected_index = st.sidebar.selectbox(
    "选择基准指数",
    options=list(bmk_options.keys()),
    index=1  # 默认选中证500
)
bmk = bmk_options[selected_index]

# 日期选择
default_start = datetime.now() - timedelta(days=365)
default_end = datetime.now()

start_date = st.sidebar.date_input(
    "开始日期",
    value=default_start,
    format="YYYY-MM-DD"
)

end_date = st.sidebar.date_input(
    "结束日期",
    value=default_end,
    format="YYYY-MM-DD"
)

# 验证日期
if start_date > end_date:
    st.sidebar.error("❌ 开始日期不能晚于结束日期！")
    st.stop()

# 显示选中的参数
st.sidebar.markdown("---")
st.sidebar.subheader("📋 当前参数")
st.sidebar.write(f"**基准指数**: {selected_index} ({bmk})")
st.sidebar.write(f"**开始日期**: {start_date}")
st.sidebar.write(f"**结束日期**: {end_date}")

# 2. 主界面
st.header("🚀 运行分析")

if st.button("开始运行", type="primary", use_container_width=True):
    st.info(f"正在分析 **{selected_index}** 从 **{start_date}** 到 **{end_date}** 的业绩归因...")
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 导入并运行分析函数
        status_text.text("正在加载分析模块...")
        progress_bar.progress(10)
        
        from weight_contribution import run_contribution_analysis
        
        status_text.text("正在运行归因分析...")
        progress_bar.progress(30)
        
        # 转换日期格式
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 运行分析
        output_path = run_contribution_analysis(bmk, start_dt, end_dt)
        
        progress_bar.progress(80)
        status_text.text("正在处理结果...")
        
        if output_path:
            st.success("✅ 分析完成！")
            
            # 读取结果文件
            interval_df = pd.read_excel(output_path, sheet_name='区间收益分析')
            interval_df = df_info[["产品名称_指数计算池","产品代码_指数计算池","授权等级"]].merge(interval_df, on="code",how="right")
            detailed_df = pd.read_excel(output_path, sheet_name='详细信息')
            
            # 保存到 session_state
            st.session_state.analysis_complete = True
            st.session_state.output_path = output_path
            st.session_state.interval_df = interval_df
            st.session_state.detailed_df = detailed_df
            
        else:
            st.warning("⚠️ 分析未完成，请检查输入参数或数据文件")
        
        progress_bar.progress(100)
        status_text.text("完成！")
        
    except Exception as e:
        st.error(f"❌ 运行出错：{str(e)}")
        st.exception(e)
        progress_bar.empty()
        status_text.empty()

# 3. 展示结果（如果分析已完成）
if st.session_state.analysis_complete:
    output_path = st.session_state.output_path
    interval_df = st.session_state.interval_df
    detailed_df = st.session_state.detailed_df
    
    st.header("📈 分析结果")
    
    # 区间收益分析
    st.subheader("区间收益分析")
    
    # 创建带色阶的样式
    styled_df = interval_df.style.format({
        '累计超额收益': '{:.4f}',
        '累计风格因子贡献': '{:.4f}',
        '累计行业因子贡献': '{:.4f}',
        '累计残差贡献': '{:.4f}',
        'PA强度': '{:.4f}'
    })
    
    # 给五列添加色阶（绿色表示好，红色表示差）
    color_columns = ['累计超额收益', '累计风格因子贡献', '累计行业因子贡献', '累计残差贡献', 'PA强度']
    for col in color_columns:
        if col in interval_df.columns:
            styled_df = styled_df.background_gradient(
                subset=[col],
                cmap='RdYlGn',  # 红-黄-绿渐变色阶
                vmin=interval_df[col].min(),
                vmax=interval_df[col].max()
            )
    
    # 检查是否有风格因子列在 interval_df 中
    style_columns = [
        '账面市值比因子累计收益',
        '非线性市值因子累计收益',
        '流动性因子累计收益',
        '盈利率因子累计收益',
        '贝塔因子累计收益',
        '规模因子累计收益',
        '动量因子累计收益',
        '杠杆率因子累计收益',
        '残余波动率因子累计收益',
        '成长因子累计收益'
    ]
    
    existing_style_columns = [col for col in style_columns if col in interval_df.columns]
    
    # 如果有风格因子列，给它们统一添加蓝色色阶
    if existing_style_columns:
        # 计算所有风格因子列的全局最小值和最大值
        all_values = []
        for col in existing_style_columns:
            all_values.extend(interval_df[col].dropna().values)
        
        global_min = min(all_values)
        global_max = max(all_values)
        
        # 统一应用色阶到所有风格因子列
        styled_df = styled_df.background_gradient(
            subset=existing_style_columns,
            cmap='Blues',  # 蓝色渐变色阶
            vmin=global_min,
            vmax=global_max
        )
    
    # 显示带色阶的表格
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # 详细信息（可折叠）
    with st.expander("📄 查看详细信息", expanded=False):
        st.dataframe(detailed_df, use_container_width=True, height=500)
    
    # 4. 下载按钮
    st.header("💾 下载结果")
    
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="📥 下载 Excel 文件",
            data=file,
            file_name=os.path.basename(output_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # 显示文件路径
    st.info(f"📁 文件已保存至：`{output_path}`")
    
    # 关键指标展示
    st.header("📊 关键指标")
    col1, col2, col3, col4 = st.columns(4)
    
    if '累计超额收益' in interval_df.columns and len(interval_df) > 0:
        avg_excess = interval_df['累计超额收益'].mean()
        col1.metric("平均累计超额收益", f"{avg_excess:.4f}")
    
    if 'PA强度' in interval_df.columns and len(interval_df) > 0:
        avg_pa = interval_df['PA强度'].mean()
        col2.metric("平均PA强度", f"{avg_pa:.4f}")
    
    if '累计风格因子贡献' in interval_df.columns and len(interval_df) > 0:
        avg_style = interval_df['累计风格因子贡献'].mean()
        col3.metric("平均风格因子贡献", f"{avg_style:.4f}")
    
    if '累计行业因子贡献' in interval_df.columns and len(interval_df) > 0:
        avg_industry = interval_df['累计行业因子贡献'].mean()
        col4.metric("平均行业因子贡献", f"{avg_industry:.4f}")
    
    # 产品详情可视化
    st.markdown("---")
    st.header("📈 产品详情可视化")
    
    # 检查是否有code列
    if 'code' in detailed_df.columns:
        unique_codes = sorted(detailed_df['code'].unique())
        
        selected_code = st.selectbox(
            "选择产品（code）",
            options=['请选择产品'] + unique_codes,
            index=0,
            key='product_code'
        )
        
        if selected_code != '请选择产品':
            # 筛选该产品的数据
            product_data = detailed_df[detailed_df['code'] == selected_code].copy()
            
            if len(product_data) > 0:
                # 显示产品数据表格
                st.subheader("产品详细数据")
                st.dataframe(product_data, use_container_width=True)
                
                # 需要的指标列
                metrics_cols = ['区间收益', '最大回撤', '年化波动', '夏普比率']
                available_metrics = [col for col in metrics_cols if col in product_data.columns]
                
                if available_metrics:
                    # 指标列都必须是数值型
                    plot_data = product_data[['指标'] + available_metrics].copy()
                    
                    # 确保指标是数值
                    for col in available_metrics:
                        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
                    
                    # 绘图
                    st.subheader(f"产品 {selected_code} 指标可视化")
                    
                    # 中文支持（设置一次即可）
                    plt.rcParams['font.sans-serif'] = ['SimHei']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 两个图并排显示
                    for i in range(0, len(available_metrics), 2):
                        cols = st.columns(2)
                        
                        # 第一个图
                        if i < len(available_metrics):
                            metric = available_metrics[i]
                            with cols[0]:
                                fig, ax = plt.subplots(figsize=(9, 5))
                                
                                # 按指标值排序
                                sorted_data = plot_data.sort_values(by=metric, ascending=True)
                                
                                # 绘制颜色：正值绿色，负值红色
                                colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in sorted_data[metric]]
                                
                                bars = ax.barh(sorted_data['指标'], sorted_data[metric], color=colors)
                                
                                # 添加数值标签
                                for bar in bars:
                                    width = bar.get_width()
                                    ax.text(
                                        width + (0.001 if width >= 0 else -0.008),
                                        bar.get_y() + bar.get_height()/2,
                                        f'{width:.4f}',
                                        va='center',
                                        fontsize=8
                                    )
                                
                                ax.set_xlabel(metric)
                                ax.set_title(f'{metric}')
                                ax.grid(axis='x', alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                        
                        # 第二个图
                        if i + 1 < len(available_metrics):
                            metric = available_metrics[i + 1]
                            with cols[1]:
                                fig, ax = plt.subplots(figsize=(9, 5))
                                
                                # 按指标值排序
                                sorted_data = plot_data.sort_values(by=metric, ascending=True)
                                
                                # 绘制颜色：正值绿色，负值红色
                                colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in sorted_data[metric]]
                                
                                bars = ax.barh(sorted_data['指标'], sorted_data[metric], color=colors)
                                
                                # 添加数值标签
                                for bar in bars:
                                    width = bar.get_width()
                                    ax.text(
                                        width + (0.001 if width >= 0 else -0.008),
                                        bar.get_y() + bar.get_height()/2,
                                        f'{width:.4f}',
                                        va='center',
                                        fontsize=8
                                    )
                                
                                ax.set_xlabel(metric)
                                ax.set_title(f'{metric}')
                                ax.grid(axis='x', alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
            else:
                st.warning("未找到该产品的数据")
    else:
        st.info("详细数据中没有code列，无法进行产品筛选")

# 5. 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 功能介绍
    本面板用于进行业绩归因分析，支持以下功能：
    
    1. **参数设置**：在侧边栏选择基准指数和日期范围
    2. **运行分析**：点击"开始运行"按钮执行归因分析
    3. **查看结果**：在网页上查看区间收益分析和详细信息
    4. **下载结果**：下载完整的 Excel 分析结果
    
    ### 支持的基准指数
    - 沪深300 (000300)
    - 中证500 (905)
    - 中证1000 (000852)
    - 国证2000 (932000)
    
    ### 输出指标
    - 累计超额收益
    - 累计风格因子贡献
    - 累计行业因子贡献
    - 累计残差贡献
    - PA强度指标
    
    ### 注意事项
    - 确保数据文件路径正确配置
    - 日期范围应在数据覆盖范围内
    - 首次运行可能需要加载数据，请耐心等待
    """)

# 页脚
st.markdown("---")
st.caption("业绩归因分析系统 | Powered by Streamlit")