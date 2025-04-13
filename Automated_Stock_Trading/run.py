from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from PPO import *
from model import *
from DRL import * 

# Access environment variables
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
API_BASE_URL = os.getenv("API_BASE_URL")
data_url = os.getenv("DATA_URL")

ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)

ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5000, "eval_gap":30,
        "eval_times":1} 
env = StockTradingEnv
load_dotenv()

train(start_date = '2022-08-25', 
      end_date = '2022-09-02',
      ticker_list = ticker_list, 
      data_source = 'alpaca',
      time_interval= '1Min', 
      technical_indicator_list= INDICATORS,
      drl_lib='elegantrl', 
      env=env, 
      model_name='ppo',
      if_vix=True, 
      API_KEY = API_KEY, 
      API_SECRET = API_SECRET, 
      API_BASE_URL = API_BASE_URL,
      erl_params=ERL_PARAMS,
      cwd='./papertrading_erl_retrain',
      break_step=2e5)

account_value_erl=test(start_date = '2022-09-01', 
                      end_date = '2022-09-02',
                      ticker_list = ticker_list, 
                      data_source = 'alpaca',
                      time_interval= '1Min', 
                      technical_indicator_list= INDICATORS,
                      drl_lib='elegantrl', 
                      env=env, 
                      model_name='ppo',
                      if_vix=True, 
                      API_KEY = API_KEY, 
                      API_SECRET = API_SECRET, 
                      API_BASE_URL = API_BASE_URL,
                      cwd='./papertrading_erl',
                      net_dimension = ERL_PARAMS['net_dimension'])



