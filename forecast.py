import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 获取数据
symbol = "9880.HK"
data = yf.download(symbol, period="6mo", interval="1d")

# 计算指标
data['MA_10'] = data['Close'].rolling(10).mean()
data['MA_30'] = data['Close'].rolling(30).mean()

# RSI
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD
data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# OBV
data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

# MTM
data['MTM'] = data['Close'] - data['Close'].shift(10)

# PSY
data['Price_Up'] = (data['Close'].diff() > 0).astype(int)
data['PSY'] = data['Price_Up'].rolling(window=12).sum() / 12 * 100

# 创建图表
fig = make_subplots(
    rows=8, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.5],
    specs=[[{"type": "xy"}],  # 第1行：xy 类型
           [{"type": "xy"}],  # 第2行：xy 类型
           [{"type": "xy"}],  # 第3行：xy 类型
           [{"type": "xy"}],  # 第4行：xy 类型
           [{"type": "xy"}],  # 第5行：xy 类型
           [{"type": "xy"}],  # 第6行：xy 类型
           [{"type": "xy"}],  # 第7行：xy 类型
           [{"type": "table"}]]  # 第8行：table 类型
)

# 添加蜡烛图
fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='价格'
    ), row=1, col=1
)

# 添加均线
for ma in ['MA_10', 'MA_30']:
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[ma],
            name=ma,
            line=dict(width=1)
        ), row=1, col=1
    )

# 添加成交量
fig.add_trace(
    go.Bar(
        x=data.index,
        y=data['Volume'],
        name='成交量',
        marker_color='rgba(100, 150, 200, 0.6)'
    ), row=2, col=1
)

# 添加RSI
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['RSI'],
        name='RSI',
        line=dict(color='purple', width=1)
    ), row=3, col=1
)

# 添加MACD
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['MACD'],
        name='MACD',
        line=dict(color='blue', width=1)
    ), row=4, col=1
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['Signal'],
        name='Signal Line',
        line=dict(color='orange', width=1)
    ), row=4, col=1
)

# 添加MACD柱状图
fig.add_trace(
    go.Bar(
        x=data.index,
        y=data['MACD'] - data['Signal'],
        name='MACD Histogram',
        marker_color=np.where(data['MACD'] - data['Signal'] > 0, 'green', 'red')
    ), row=4, col=1
)

# 添加OBV
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['OBV'],
        name='OBV',
        line=dict(color='cyan', width=1)
    ), row=5, col=1
)

# 添加MTM
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['MTM'],
        name='MTM',
        line=dict(color='magenta', width=1)
    ), row=6, col=1
)

# 添加PSY
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['PSY'],
        name='PSY',
        line=dict(color='yellow', width=1)
    ), row=7, col=1
)

# 生成买卖信号表格
signals = []

for i in range(len(data)):
    date = data.index[i]  # 使用 .index 获取日期
    close_price = data['Close'].iloc[i]  # 使用 .iloc 基于位置索引
    rsi = data['RSI'].iloc[i]
    macd = data['MACD'].iloc[i]
    signal_line = data['Signal'].iloc[i]
    obv = data['OBV'].iloc[i]
    mtm = data['MTM'].iloc[i]

    # RSI 信号
    if rsi > 70:
        signals.append([date, "卖出", close_price, "RSI 超买 >70"])
    elif rsi < 30:
        signals.append([date, "买入", close_price, "RSI 超卖 <30"])

    # MACD 信号
    if i > 0:  # 确保 i-1 是有效的索引
        if macd > signal_line and data['MACD'].iloc[i-1] <= data['Signal'].iloc[i-1]:
            signals.append([date, "买入", close_price, "MACD 金叉"])
        elif macd < signal_line and data['MACD'].iloc[i-1] >= data['Signal'].iloc[i-1]:
            signals.append([date, "卖出", close_price, "MACD 死叉"])

    # OBV 信号
    if i > 0 and obv > data['OBV'].iloc[i-1]:
        signals.append([date, "买入", close_price, "OBV 上升，资金流入"])
    elif i > 0 and obv < data['OBV'].iloc[i-1]:
        signals.append([date, "卖出", close_price, "OBV 下降，资金流出"])

    # MTM 信号
    if mtm > 0:
        signals.append([date, "买入", close_price, "MTM 正值，价格上涨动量"])
    elif mtm < 0:
        signals.append([date, "卖出", close_price, "MTM 负值，价格下跌动量"])

# 将信号转换为DataFrame
signals_df = pd.DataFrame(signals, columns=["日期", "操作", "价格", "信号类型"])

# 创建表格
signal_table = go.Table(
    header=dict(
        values=["日期", "操作", "价格", "信号类型"],
        font=dict(size=14, color='white', family="Arial"),  # 增大表头字体
        fill_color='navy',
        align='center',
        height=40  # 增加表头高度
    ),
    cells=dict(
        values=[signals_df["日期"], signals_df["操作"], signals_df["价格"], signals_df["信号类型"]],
        font=dict(size=12, color='black', family="Arial"),  # 增大单元格字体
        fill_color=[['rgb(240, 240, 240)', 'rgb(220, 220, 220)'] * len(signals_df)],  # 交替行背景色
        align='center',
        height=30  # 增加单元格高度
    ),
    columnwidth=[200, 100, 100, 300]  # 调整列宽
)

# 将表格添加到图表中
fig.add_trace(signal_table, row=8, col=1)  # 将表格添加到第8行

# 布局设置
fig.update_layout(
    title=f"{symbol} 技术分析与交易信号",
    template="plotly_dark",
    hovermode="x unified",
    height=1400,  # 增加高度以适应表格
    showlegend=True,
    margin=dict(l=50, r=50, b=50, t=100),  # 调整边距
)

# 坐标轴标签
fig.update_yaxes(title_text="价格 (HKD)", row=1, col=1)
fig.update_yaxes(title_text="成交量", row=2, col=1)
fig.update_yaxes(title_text="RSI", range=[0,100], row=3, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)
fig.update_yaxes(title_text="OBV", row=5, col=1)
fig.update_yaxes(title_text="MTM", row=6, col=1)
fig.update_yaxes(title_text="PSY", range=[0,100], row=7, col=1)

# 表格的坐标轴设置
fig.update_yaxes(title_text="交易信号", row=8, col=1, showticklabels=False)  # 隐藏表格的Y轴刻度
fig.update_xaxes(title_text="日期", row=8, col=1)  # 添加X轴标签

# 定义 CSS 样式
custom_css = """
<style>
/* 整体页面样式 */
body {
    font-family: Arial, sans-serif;
    background-color: #1e1e1e; /* 深色背景 */
    color: #ffffff; /* 白色文字 */
    margin: 0;
    padding: 20px;
}

/* 图表容器样式 */
.plotly-graph-div {
    background-color: #2a2a2a; /* 图表背景色 */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

/* 表格容器样式 */
.table-container {
    max-height: 400px; /* 固定表格高度 */
    overflow-y: auto; /* 添加垂直滚动条 */
    background-color: #2a2a2a; /* 表格背景色 */
    border-radius: 10px;
    padding: 10px;
    margin-top: 20px;
}

/* 表格样式 */
table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
}

table th, table td {
    padding: 12px;
    text-align: center;
    border: 1px solid #444; /* 表格边框 */
}

table th {
    background-color: #333; /* 表头背景色 */
    color: #ffffff; /* 表头文字颜色 */
    font-size: 14px;
    font-weight: bold;
}

table td {
    background-color: #2a2a2a; /* 单元格背景色 */
    color: #ffffff; /* 单元格文字颜色 */
    font-size: 12px;
}

/* 交替行背景色 */
table tr:nth-child(even) td {
    background-color: #333; /* 偶数行背景色 */
}

/* 滚动条样式 */
.table-container::-webkit-scrollbar {
    width: 8px;
}

.table-container::-webkit-scrollbar-track {
    background: #2a2a2a; /* 滚动条轨道颜色 */
    border-radius: 4px;
}

.table-container::-webkit-scrollbar-thumb {
    background: #666; /* 滚动条滑块颜色 */
    border-radius: 4px;
}

.table-container::-webkit-scrollbar-thumb:hover {
    background: #888; /* 滚动条滑块悬停颜色 */
}
</style>
"""

# 保存为 HTML 文件
fig.write_html(
    "stock_analysis_with_signals.html",
    include_plotlyjs='cdn',
    config={"scrollZoom": True},  # 允许缩放
    full_html=True,  # 生成完整 HTML 文件
    div_id="chart-container"  # 图表容器的 ID
)

# 将 CSS 插入到生成的 HTML 文件中
with open("stock_analysis_with_signals.html", "r+", encoding="utf-8") as file:
    content = file.read()
    file.seek(0, 0)
    file.write(custom_css + content)

print("HTML 文件已保存为 stock_analysis_with_signals.html")