# RSI-MACD-Trading-Bot
A trading bot that generates buy and sell signals based on RSI and MACD. Uses YFinance for price data and plots backtests on interactive graphs. Can be modified to connect to brokerage accounts to automatize trading.

Important notes

    Input ticker-symbol must be same as appears on yahoo finance (Currently set to QQQ)
    Time-Interval must be a year value

Disclaimer

    Trading bot may not always be profitable and may cause loss of money. Use at your own risk in personal trading.

Setup

    Make sure you have all the libraries that need to be imported for the main.py file. If not, run the following in your terminal:
    pip3 install yfinance
    pip3 install pandas
    pip3 install plotly
    pip3 install numpy

How to use

    Open file and imput ticker symbol wanting to backtest strategy on 
      For example: stock_symbol = "QQQ"
    Under genrate signals code you can delete # in code to give a sell signal on last day if open buy signal so you get an accurate profit percentage currently
      For example: 
      if positions > 0:
        sell_signals.append(price.index[-1])
    Run file and it will display information on three interactive plotly graphs


