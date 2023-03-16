# Momentum Strategy Implementation

import backtrader as bt
import numpy as np
from scipy.stats import linregress
import pandas as pd


class Momentum(bt.Indicator):
    # every trading day had a momentum parameter
    # except for the first 90 days
    lines = ('momentum_trend',)
    params = (('period', 90),)

    def __init__(self):
        self.addminperiod(self.params.period)

    '''
    Calculates and sets the momentum trend  (first 90 days will have no momentum since we look 
    90 days in the past to calculate the trend)
    '''
    def next(self):
        returns = np.log(self.data.get(size=self.params.period))
        x = np.arange(len(returns))
        beta, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + beta) ** 252
        # [0] is the actual value returns and [-1] would be the previous day
        self.lines.momentum_trend[0] = annualized * (rvalue**2)


# This is a Backtrader Strategy
class MomentumStrategy(bt.Strategy):

    def __init__(self):
        self.counter = 0
        self.indicators = {}
        self.sorted_data = []
        # we store the S&P500 (the index) data as the first item of the dataset
        self.spy = self.datas[0]
        # all the other stocks (present in S&P500)
        self.stocks = self.datas[1:]

        for stock in self.stocks:
            self.indicators[stock] = {}
            self.indicators[stock]['momentum'] = Momentum(stock.close, period=90)
            self.indicators[stock]['sma100'] = bt.indicators.SimpleMovingAverage(stock.close, period=100)
            self.indicators[stock]['atr20'] = bt.indicators.ATR(stock, period=20)

        # SMA for S&P500 index - because we open long positions when the S&P500 index is > SMA(200) [Bullish market]
        self.sma200 = bt.indicators.MovingAverageSimple(self.spy.close, period=200)

    def prenext(self):
        # count number of days elapsed
        self.next()

    '''
    Set Day Counter to keep portfolio balanced every week
    We also want to update the position every 2 weeks
    '''
    def next(self):
        # a week has passed sp we have to make trades if needed
        if self.counter % 5 == 0:
            self.rebalance_portfolio()
        if self.counter % 10 == 0:
            # 2 weeks have passed
            self.update_positions()

        self.counter += 1

    '''
    Rebalancing the portfolio (Done every week -- 5 trading days)
    Top 20% threshold set for buying/selling stocks 
    Also check the simple moving average of the past 100 days and if the stock falls below that we sell
    '''
    def rebalance_portfolio(self):

        self.sorted_data = list(filter(lambda stock_data: len(stock_data) > 100, self.stocks))
        # sort the S&P500 stocks based on momentum
        self.sorted_data.sort(key=lambda stock: self.indicators[stock]['momentum'][0])
        num_stocks = len(self.sorted_data)

        # sell stocks (close the long positions) - top 20%
        for index, single_stock in enumerate(self.sorted_data):
            # we can check whether are there any open positions
            if self.getposition(self.data).size:
                # if the stock is not in the top 20% then close the long position
                # if the stock price falls below the 100 day MA sell the stock
                if index > 0.2 * num_stocks or single_stock < self.indicators[single_stock['sma100']]:
                    self.close()

        # we open long positions when SMA is below the S&P500 index
        if self.spy < self.sma200:
            return

        # open long positions - consider the top 20% of the stocks and buy accordingly
        for index, single_stock in enumerate(self.sorted_data[:int(num_stocks * 0.2)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash > 0 and not self.getposition(self.data).size:
                # 0.001 is chosen risk factor
                size = value * 0.001 / self.indicators[single_stock]['atr20']
                self.buy(single_stock, size=size)

    '''
    Update the position (Done every 2 week -- 10 trading days)
    Buy stocks based on the top 20% momentum
    and use Average True Range (ATR) indicator to calculate size
    '''
    def update_positions(self):
        num_stocks = len(self.sorted_data)

        # top 20% momentum range
        for index, single_stock in enumerate(self.sorted_data[:int(0.2 * num_stocks)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash > 0:
                # 0.001 is the chosen risk factor
                size = value * 0.001 / self.indicators[single_stock]["atr20"]
                self.order_target_size(single_stock, size)


if __name__ == '__main__':

    # Using Backtrader ti implement this strategy
    # https://www.backtrader.com/docu/

    stocks = []
    # Initialize BackTrader Cerebro
    cerebro = bt.Cerebro()

    with open("Data/companies_all") as file_in:
        for line in file_in:
            stocks.append(line.strip('\n'))
            try:
                df = pd.read_csv('Data/{}'.format(line.strip('\n')), parse_dates=True, index_col=0)
                if len(df) > 100:
                    # Attach Data
                    cerebro.adddata(bt.feeds.PandasData(dataname=df))
            except FileNotFoundError:
                pass

    # Analytics Tracker
    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    # Attach Strategy
    cerebro.addstrategy(MomentumStrategy)

    cerebro.broker.set_cash(100000)
    # commission fee is 1%
    cerebro.broker.setcommission(0.01)

    print('Initial capital: $%.2f' % cerebro.broker.getvalue())
    # Run Strategy
    results = cerebro.run()

    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    print('Final Capital: $%.2f' % cerebro.broker.getvalue())
