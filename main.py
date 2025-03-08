import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fetch_stock_data(symbol, period="1mo", interval="1d"):
    """获取股票历史数据"""
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False)

    if data.empty:
        raise ValueError("未能获取到数据，请检查股票代码或网络连接。")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    return data


def calculate_indicators(data):
    """计算 RSI、MACD、OBV、MTM 指标"""
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # ✅ 计算 RSI 指标
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ✅ 计算 MACD 指标
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # ✅ 计算 OBV 指标
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

    # ✅ 计算 MTM（动量指标）
    data['MTM'] = data['Close'] - data['Close'].shift(10)

    # ✅ 新增 PSY 指标（12日心理线）
    data['Price_Up'] = (data['Close'].diff() > 0).astype(int)  # 当日上涨为1
    data['PSY'] = data['Price_Up'].rolling(window=12).sum() / 12 * 100  # 12日PSY

    data.bfill(inplace=True)  # 处理 NaN 值
    return data


def generate_psy_signals(data):
    """✅ 新增：基于 PSY 生成交易信号"""
    signals = []
    for i in range(1, len(data)):
        date = data.index[i]
        close_price = data['Close'].iloc[i]
        psy = data['PSY'].iloc[i]

        signal = None
        if psy < 25:
            signal = "买入（PSY 超卖 <25）"
        elif psy > 75:
            signal = "卖出（PSY 超买 >75）"

        if signal:
            signals.append(f"{date.date()} - {signal}，收盘价：{close_price:.2f}")
    return signals


def generate_rsi_signals(data):
    """基于 RSI 生成交易信号"""
    signals = []
    for i in range(1, len(data)):
        date = data.index[i]
        close_price = data['Close'].iloc[i]
        rsi = data['RSI'].iloc[i]

        signal = None
        if rsi < 30:
            signal = "买入（RSI 超卖 <30）"
        elif rsi > 70:
            signal = "卖出（RSI 超买 >70）"

        if signal:
            signals.append(f"{date.date()} - {signal}，收盘价：{close_price:.2f}")
    return signals


def generate_macd_signals(data):
    """基于 MACD 生成交易信号"""
    signals = []
    for i in range(1, len(data)):
        date = data.index[i]
        close_price = data['Close'].iloc[i]
        macd, signal_line = data['MACD'].iloc[i], data['Signal_Line'].iloc[i]

        signal = None
        if macd > signal_line and data['MACD'].iloc[i - 1] <= data['Signal_Line'].iloc[i - 1]:
            signal = "买入（MACD 金叉）"
        elif macd < signal_line and data['MACD'].iloc[i - 1] >= data['Signal_Line'].iloc[i - 1]:
            signal = "卖出（MACD 死叉）"

        if signal:
            signals.append(f"{date.date()} - {signal}，收盘价：{close_price:.2f}")
    return signals


def generate_obv_signals(data):
    """基于 OBV 生成交易信号"""
    signals = []
    for i in range(1, len(data)):
        date = data.index[i]
        close_price = data['Close'].iloc[i]
        obv = data['OBV'].iloc[i]

        signal = None
        if obv > data['OBV'].iloc[i - 1]:
            signal = "买入（OBV 上升，资金流入）"
        elif obv < data['OBV'].iloc[i - 1]:
            signal = "卖出（OBV 下降，资金流出）"

        if signal:
            signals.append(f"{date.date()} - {signal}，收盘价：{close_price:.2f}")
    return signals


def generate_mtm_signals(data):
    """基于 MTM 生成交易信号"""
    signals = []
    for i in range(1, len(data)):
        date = data.index[i]
        close_price = data['Close'].iloc[i]
        mtm = data['MTM'].iloc[i]

        signal = None
        if mtm > 0:
            signal = "买入（MTM 正值，价格上涨动量）"
        elif mtm < 0:
            signal = "卖出（MTM 负值，价格下跌动量）"

        if signal:
            signals.append(f"{date.date()} - {signal}，收盘价：{close_price:.2f}")
    return signals


def monte_carlo_simulation(data, days=30, simulations=1000):
    """使用蒙特卡洛方法预测未来股价"""
    last_price = data['Close'].iloc[-1]
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

    mean_return = log_returns.mean()
    std_return = log_returns.std()

    price_matrix = np.zeros((days, simulations))
    price_matrix[0] = last_price

    for t in range(1, days):
        random_shocks = np.random.normal(mean_return, std_return, simulations)
        price_matrix[t] = price_matrix[t - 1] * np.exp(random_shocks)

    return price_matrix


class StockVisualizer:
    """股票数据可视化工具类"""

    def __init__(self, data, signals, forecast):
        """
        初始化可视化工具
        :param data: 包含指标数据的 DataFrame
        :param signals: 合并后的交易信号列表
        :param forecast: 蒙特卡洛模拟结果矩阵
        """
        self.data = data
        self.signals = signals
        self.forecast = forecast
        self.figures = []

    def plot_price_signals(self):
        """绘制价格走势与交易信号"""
        fig, ax = plt.subplots(figsize=(14, 6))

        # 价格曲线
        ax.plot(self.data['Close'], label='收盘价', color='#1f77b4', lw=2)
        ax.plot(self.data['SMA_50'], label='50日均线', color='orange', ls='--', lw=1)
        ax.plot(self.data['SMA_200'], label='200日均线', color='green', ls='--', lw=1)

        # 标记信号
        buy_dates, buy_prices = self._extract_signal_points('买入')
        sell_dates, sell_prices = self._extract_signal_points('卖出')

        ax.scatter(buy_dates, buy_prices, color='limegreen', s=100,
                   edgecolor='darkgreen', marker='^', label='买入信号')
        ax.scatter(sell_dates, sell_prices, color='tomato', s=100,
                   edgecolor='darkred', marker='v', label='卖出信号')

        ax.set_title("价格走势与交易信号")
        ax.set_ylabel("价格 (HKD)")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        self.figures.append(fig)
        return fig

    def plot_oscillators(self):
        """绘制震荡指标 (RSI + PSY)"""
        fig, ax = plt.subplots(figsize=(14, 4))

        # RSI
        ax.plot(self.data['RSI'], label='RSI', color='purple', lw=1)
        ax.fill_between(self.data.index, 70, 100, color='red', alpha=0.1)
        ax.fill_between(self.data.index, 0, 30, color='green', alpha=0.1)

        # PSY
        ax.plot(self.data['PSY'], label='PSY (12日)', color='teal', ls='--', lw=1)

        # 参考线
        ax.axhline(25, color='teal', ls=':', alpha=0.5, label='PSY 超卖线')
        ax.axhline(75, color='teal', ls=':', alpha=0.5, label='PSY 超买线')
        ax.axhline(30, color='gray', ls='--', lw=0.8, label='RSI 超卖线')
        ax.axhline(70, color='gray', ls='--', lw=0.8, label='RSI 超买线')

        ax.set_ylabel("RSI & PSY")
        ax.set_ylim(0, 100)
        ax.legend(ncol=2, fontsize=8)
        ax.grid(alpha=0.3)
        self.figures.append(fig)
        return fig

    def plot_macd(self):
        """绘制MACD指标"""
        fig, ax = plt.subplots(figsize=(14, 4))

        ax.plot(self.data['MACD'], label='MACD', color='blue', lw=1)
        ax.plot(self.data['Signal_Line'], label='信号线', color='orange', lw=1)
        ax.bar(self.data.index,
               self.data['MACD'] - self.data['Signal_Line'],
               color=np.where(self.data['MACD'] > self.data['Signal_Line'], 'limegreen', 'tomato'),
               alpha=0.3, width=0.8)
        ax.axhline(0, color='gray', ls='--', lw=0.8)

        ax.set_ylabel("MACD")
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        self.figures.append(fig)
        return fig

    def plot_monte_carlo(self):
        """绘制蒙特卡洛模拟结果"""
        if self.forecast is None:
            print("⚠️ 无蒙特卡洛模拟数据")
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(self.forecast, color='steelblue', alpha=0.05)
        ax.plot(np.median(self.forecast, axis=1), color='darkorange',
                lw=2, label='中位值')
        ax.fill_between(
            range(30),
            np.percentile(self.forecast, 5, axis=1),
            np.percentile(self.forecast, 95, axis=1),
            color='skyblue', alpha=0.3, label='90% 置信区间'
        )

        ax.set_title("蒙特卡洛模拟 - 未来30天价格预测")
        ax.set_xlabel("交易日")
        ax.set_ylabel("预测价格 (HKD)")
        ax.legend()
        ax.grid(alpha=0.3)
        self.figures.append(fig)
        return fig

    def show_all(self):
        """显示所有图表"""
        plt.show()

    def _extract_signal_points(self, signal_type):
        """提取指定类型信号的日期和价格"""
        dates = []
        prices = []
        for s in self.signals:
            if signal_type in s:
                date_str = s.split(" - ")[0]
                try:
                    price = float(s.split("收盘价：")[1])
                    dates.append(pd.to_datetime(date_str))
                    prices.append(price)
                except (IndexError, ValueError):
                    continue  # 忽略格式错误的信号
        return dates, prices

def main():
    symbol = "9880.HK"
    data = fetch_stock_data(symbol, period="6mo")
    data = calculate_indicators(data)

    # 生成信号
    signals = {
        'rsi': generate_rsi_signals(data),
        'macd': generate_macd_signals(data),
        'obv': generate_obv_signals(data),
        'mtm': generate_mtm_signals(data),
        'psy': generate_psy_signals(data)
    }
    all_signals = sorted(
        [s for sublist in signals.values() for s in sublist],
        key=lambda x: pd.to_datetime(x.split(" - ")[0])  # ✅ 修复括号闭合
    )

    # 蒙特卡洛模拟
    forecast = monte_carlo_simulation(data)

    # 使用可视化类
    visualizer = StockVisualizer(data, all_signals, forecast)
    visualizer.plot_price_signals()
    visualizer.plot_oscillators()
    visualizer.plot_macd()
    visualizer.plot_monte_carlo()

    # 信号输出
    print("\n🔔 综合交易信号分析")
    print(tabulate(
        [[s.split(" - ")[0],
         "买入" if "买入" in s else "卖出",
         s.split("，收盘价：")[1],
         s.split("（")[1].split("）")[0]]
        for s in all_signals
    ], headers=["日期", "操作", "价格", "信号类型"], tablefmt="pretty"))  # ✅ 修复参数传递

    print("\n📊 信号统计摘要:")
    for k, v in signals.items():
        print(f"• {k.upper()} 信号: {len(v)} 个")

    visualizer.show_all()

if __name__ == "__main__":
    main()

    # 高级用法示例
    # data = fetch_stock_data("0700.HK", period="1y")
    # data = calculate_indicators(data)
    # vis = StockVisualizer(data, [], None)
    # vis.plot_price_signals()
    # vis.plot_oscillators()
    # vis.show_all()