import yfinance as yf
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np

# Download data
stock_symbol = "QQQ"
stock = yf.download(stock_symbol, period="1y")
price = stock["Adj Close"].round(2)

# Calculate YTD percentage change
start_price = price.iloc[0]
end_price = price.iloc[-1]
ytd = ((end_price - start_price) / start_price) * 100

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs)).round(2)
    return rsi

# Function to calculate MACD
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

#Generate Signals
def generate_signals(price, rsi, macd_line, signal_line, rsi_buy_threshold=30, rsi_sell_threshold=70):
    buy_signals = []
    sell_signals = []
    positions = 0

    for i in range(1, len(price)):
        if (rsi[i] <= rsi_buy_threshold or macd_line[i] > signal_line[i]) and positions == 0:
            buy_signals.append(price.index[i])
            positions += 1
        elif (rsi[i] >= rsi_sell_threshold and macd_line[i] < signal_line[i]) and positions > 0:
            sell_signals.append(price.index[i])
            positions = 0

    # Generate a sell signal on the last day (Do not use in trading - just for display)
    #if positions > 0:
        #sell_signals.append(price.index[-1])
    
    return buy_signals, sell_signals

#Generate combined RSI and MACD signals
rsi = calculate_rsi(price)
macd_line, signal_line, macd_histogram = calculate_macd(price)
buy_signals_combined, sell_signals_combined = generate_signals(price, rsi, macd_line, signal_line)

#Percent Increase Calculator
def calculate_percent_increase(data, buy_signals, sell_signals):
    total_increase = 0
    for buy_date, sell_date in zip(buy_signals, sell_signals):
        buy_price = data.loc[buy_date]
        sell_price = data.loc[sell_date]
        increase = ((sell_price - buy_price) / buy_price) * 100
        total_increase += increase
    return total_increase

percent_increase_combined = calculate_percent_increase(price, buy_signals_combined, sell_signals_combined)

# Plotting
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.075,
                    subplot_titles=(f"{stock_symbol} Buy & Sell Signals", "RSI", "MACD"), row_heights=[2/3, 0.5/3, 0.5/3])

# QQQ upper
fig.add_trace(go.Scatter(x=price.index, y=price, mode='lines', name=f"{stock_symbol} Price", legend="legend1"), row=1, col=1)

# Buy/Sell Signals for RSI and MACD
fig.add_trace(go.Scatter(x=buy_signals_combined, y=price.loc[buy_signals_combined], mode='markers', name='Buy (RSI & MACD)', marker=dict(color='green', size=8), legend="legend1"), row=1, col=1)
fig.add_trace(go.Scatter(x=sell_signals_combined, y=price.loc[sell_signals_combined], mode='markers', name='Sell (RSI & MACD)', marker=dict(color='red', size=8), legend="legend1"), row=1, col=1)

# RSI Lower
fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines', name="RSI", line=dict(color="orange"), legendgroup='RSI Graph', legend='legend2'), row=2, col=1)
fig.add_shape(go.layout.Shape(type='line', x0=rsi.index[0], x1=rsi.index[-1], y0=30, y1=30, line=dict(color='green', dash='dash'), name='RSI = 30', legendgroup='RSI Graph', legend='legend2', showlegend=True), row=2, col=1)
fig.add_shape(go.layout.Shape(type='line', x0=rsi.index[0], x1=rsi.index[-1], y0=70, y1=70, line=dict(color='red', dash='dash'), name='RSI = 70', legendgroup='RSI Graph', legend='legend2', showlegend=True), row=2, col=1)

# MACD Upper
fig.add_trace(go.Scatter(x=price.index, y=macd_line, mode='lines', name="MACD Line", line=dict(color="blue"), legendgroup='MACD Graph', legend="legend3"), row=3, col=1)
fig.add_trace(go.Scatter(x=price.index, y=signal_line, mode='lines', name="Signal Line", line=dict(color="red"), legendgroup='MACD Graph', legend="legend3"), row=3, col=1)
fig.add_trace(go.Bar(x=price.index, y=macd_histogram, name="MACD Histogram", marker=dict(color='gray'), legendgroup='MACD Graph', legend="legend3"), row=3, col=1)

# Update layout
fig.update_layout(title_text="RSI & MACD Signals", title_font=dict(size=30), yaxis_title='Price',showlegend=True,
legend1=dict(x=1.02,y=1,traceorder='normal',bgcolor='rgba(255, 255, 255, 0.5)',bordercolor='rgba(0, 0, 0, 0.2)',borderwidth=1,tracegroupgap=10,title_text=f'{stock_symbol} Graph'),
legend2=dict(x=1.02,y=0.21,traceorder='normal',bgcolor='rgba(255, 255, 255, 0.5)',bordercolor='rgba(0, 0, 0, 0.2)',borderwidth=1,tracegroupgap=10,title_text='RSI Graph'),
legend3=dict(x=1.02,y=0.001,traceorder='normal',bgcolor='rgba(255, 255, 255, 0.5)',bordercolor='rgba(0, 0, 0, 0.2)',borderwidth=1,tracegroupgap=10,title_text='MACD Graph'))

fig.show()

max_y_val = max(ytd, percent_increase_combined)

# Bar graph
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(x=['1 Yr %', 'Strategy %'],y=[ytd, percent_increase_combined],marker_color = ['blue', 'green'] if percent_increase_combined > ytd else (['blue', 'blue'] if percent_increase_combined == ytd else ['blue', 'red']),hovertemplate='<b></b>%{y:.2f}%',name='',showlegend=False,))
fig_bar.add_trace(go.Bar(x=[None],y=[None],marker=dict(color='blue'),hoverinfo='none',showlegend=True,name=f'1 Yr: {ytd:.2f}%'))
fig_bar.add_trace(go.Bar(x=[None],y=[None],marker=dict(color='green' if percent_increase_combined > ytd else ('blue' if percent_increase_combined == ytd else 'red')),hoverinfo='none',showlegend=True,name=f'Strategy: {percent_increase_combined:.2f}%'))

#Displays price on bar
for i, y_val in enumerate([ytd, percent_increase_combined]):
    annotation_y_pos = max_y_val * 0.98
    fig_bar.add_annotation(x=i,y=y_val,text=f"{y_val:.2f}%",showarrow=False,font=dict(color="white", size=12),ax=0,yshift=-10)

fig_bar.update_layout(title='Stategy vs. 1Yr Percent',title_font=dict(size=30),xaxis=dict(tickfont=dict(size=20)), yaxis_title='Percent Change',barmode='group',showlegend=True,legend=dict(x=1.02,y=1.00,traceorder='normal',bgcolor='rgba(255, 255, 255, 0.5)',bordercolor='rgba(0, 0, 0, 0.2)',borderwidth=1,tracegroupgap=10,title=dict(text='Percent Change')))

fig_bar.show()

#Display Trades 
signals_list = []

for buy_date in buy_signals_combined:
    signals_list.append({'Date': buy_date, 'Price': price.loc[buy_date], 'Signal': 'Buy'})

for sell_date in sell_signals_combined:
    signals_list.append({'Date': sell_date, 'Price': price.loc[sell_date], 'Signal': 'Sell'})

signals_df = pd.DataFrame(signals_list)

signals_df = signals_df.sort_values(by='Date', ascending=False).reset_index(drop=True)

signals_df['P/L'] = np.where(signals_df['Signal'] == 'Sell', -(signals_df['Price'].shift(-1) - signals_df['Price']).round(2), '')
signals_df['P/L %'] = np.where(signals_df['Signal'] == 'Sell', ((-(signals_df['Price'].shift(-1) - signals_df['Price']) / signals_df['Price']) * 100).round(2).astype(str) + '%', '')

table = ff.create_table(signals_df)

table.show()