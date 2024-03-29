# RSI Trading Implementation

import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd


class MovingAverageRSIStrategy:

    def __init__(self, capital, stock, start, end, short_period, long_period):
        self.data = None
        self.is_long = False
        self.short_period = short_period
        self.long_period = long_period
        self.capital = capital
        self.equity = [capital]
        self.stock = stock
        self.start = start
        self.end = end

    '''
    Function to get a specific stock based on started and end date
    This returns a data frame containing adjusted closing prices for the specific chosen stock
    '''
    def download_data(self):
        stock_data = {}
        ticker = yf.download(self.stock, self.start, self.end)
        stock_data['price'] = ticker['Adj Close']
        self.data = pd.DataFrame(stock_data)

    '''
    Create various new columns with specific signals needed for strategy
    '''
    def construct_signals(self):
        self.data['short_ma'] = self.data['price'].ewm(span=self.short_period).mean()
        self.data['long_ma'] = self.data['price'].ewm(span=self.long_period).mean()
        self.data['move'] = self.data['price'] - self.data['price'].shift(1)
        self.data['up'] = np.where(self.data['move'] > 0, self.data['move'], 0)
        self.data['down'] = np.where(self.data['move'] < 0, self.data['move'], 0)
        self.data['average_gain'] = self.data['up'].rolling(14).mean()
        self.data['average_loss'] = self.data['down'].abs().rolling(14).mean()
        relative_strength = self.data['average_gain'] / self.data['average_loss']
        self.data['rsi'] = 100.0 - (100.0 / (1.0 + relative_strength))
        self.data = self.data.dropna()
        print(self.data)

    '''
    Plotting all of the constructed signals 
    '''
    def plot_signals(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['price'], label='Stock Price', color='grey')
        plt.plot(self.data['short_ma'], label='Short MA', color='blue')
        plt.plot(self.data['long_ma'], label='Long MA', color='green')
        plt.title('IBM - Moving Average Crossover Trading Strategy with RSI')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.show()

    '''
    Simulate Buying and Selling points based on moving crossover strategy
    If Short MA is above Long MA buy since this is an indicator of a bullish market
    If Short MA is below Long MA sell since this is an indicator of a bearish market
    Also taking into account if RSI < 30 since that would indicate a oversold
    Whereas a RSI > 70 would indicate a overbought asset
    '''
    def simulate(self):
        # Consider all of the trading days and decide whether to open
        # a long position or not
        price_when_buy = 0

        for index, row in self.data.iterrows():
            # Close the long position we have opened
            if row['short_ma'] < row['long_ma'] and self.is_long:
                self.equity.append(self.capital * row.price / price_when_buy)
                self.is_long = False
                # SELLING
            elif row['short_ma'] > row['long_ma'] and not self.is_long and row['rsi'] < 30:
                # Open a long position
                price_when_buy = row['price']
                self.is_long = True
                # BUYING

    '''
    Plot the Equity Curve
    '''
    def plot_equity(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity, label='Stock Price', color='green')
        plt.title('IBM - Moving Average Crossover Strategy with RSI Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Actual Capital ($)')
        plt.show()

    '''
    Calculate Metrics (Profit, Capital, and Sharpe Ratio)
    '''
    def show_stats(self):
        print("Profit of the trading strategy: %.2f%%" % (
            (float(self.equity[-1]) - float(self.equity[0])) /
            float(self.equity[0]) * 100))
        print("Actual capital: $%0.2f" % self.equity[-1])
        returns = (self.data['price'] - self.data['price'].shift(1)) / self.data['price'].shift(1)
        # There are 252 trading days per year and we are looking at annual rate
        ratio = returns.mean() / returns.std() * np.sqrt(252)
        print('Sharpe ratio: %.2f' % ratio)


if __name__ == '__main__':
    # Date range
    start_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2020, 1, 1)

    initial_capital = 100
    stock = 'IBM'
    # The span of the days the short and long position will be calculated based on respectively
    short_period = 40
    long_period = 150

    model = MovingAverageRSIStrategy(initial_capital, stock, start_date, end_date, short_period, long_period)
    model.download_data()
    model.construct_signals()
    model.plot_signals()
    model.simulate()
    model.plot_equity()
    model.show_stats()
