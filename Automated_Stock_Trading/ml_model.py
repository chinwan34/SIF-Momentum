import pandas as pd
import numpy as np
import traceback
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV,RandomizedSearchCV
import time
import os
import errno

from multiprocessing import cpu_count

n_cpus = cpu_count() - 1
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def prepare_rolling_train(df, features_column, label_column, date_column, unique_datetime, testing_windows, first_trade_date_index, max_rolling_window_index, current_index):
    """
    Prepares training data for rolling window analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data
        features_column (list): List of feature column names
        label_column (str): Name of the label column
        date_column (str): Name of the date column
        unique_datetime (list): List of unique dates
        testing_windows (int): Number of testing windows
        first_trade_date_index (int): Index of first trade date
        max_rolling_window_index (int): Maximum rolling window index
        current_index (int): Current index in the rolling window
        
    Returns:
        tuple: (X_train, y_train) containing training features and labels
    """
    if current_index <=max_rolling_window_index:
        train=df[(df[date_column] >= unique_datetime[0]) \
                & (df[date_column] < unique_datetime[current_index-testing_windows])]
    else:
        train=df[(df[date_column] >= unique_datetime[current_index-max_rolling_window_index]) \
                & (df[date_column] < unique_datetime[current_index-testing_windows])]
        
    X_train=train[features_column]
    y_train=train[label_column]
    print(X_train.shape, y_train.shape, testing_windows, first_trade_date_index, unique_datetime, current_index)
    return X_train,y_train

def prepare_rolling_test(df, features_column, label_column, date_column, unique_datetime, testing_windows, fist_trade_date_index, current_index):
    """
    Prepares test data for rolling window analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data
        features_column (list): List of feature column names
        label_column (str): Name of the label column
        date_column (str): Name of the date column
        unique_datetime (list): List of unique dates
        testing_windows (int): Number of testing windows
        fist_trade_date_index (int): Index of first trade date
        current_index (int): Current index in the rolling window
        
    Returns:
        tuple: (X_test, y_test) containing test features and labels
    """
    test=df[(df[date_column] >= unique_datetime[current_index-testing_windows]) \
            & (df[date_column] < unique_datetime[current_index])]
    X_test=test[features_column]
    y_test=test[label_column]
    return X_test,y_test

def prepare_trade_data(df, features_column, label_column, date_column, tic_column, unique_datetime, testing_windows, fist_trade_date_index, current_index):
    """
    Prepares trading data for a specific date.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data
        features_column (list): List of feature column names
        label_column (str): Name of the label column
        date_column (str): Name of the date column
        tic_column (str): Name of the ticker column
        unique_datetime (list): List of unique dates
        testing_windows (int): Number of testing windows
        fist_trade_date_index (int): Index of first trade date
        current_index (int): Current index in the rolling window
        
    Returns:
        tuple: (X_trade, y_trade, trade_tic) containing trading features, labels and tickers
    """
    trade  = df[df[date_column] == unique_datetime[current_index]]
    X_trade = trade[features_column]
    y_trade = trade[label_column]
    trade_tic = trade[tic_column].values
    return X_trade,y_trade,trade_tic

def train_linear_regression(X_train, y_train):
    """
    Trains a linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        LinearRegression: Trained linear regression model
    """
    lr_regressor = LinearRegression()
    model = lr_regressor.fit(X_train, y_train)
    return model

def train_recursive_feature_elimination(X_train, y_train):
    """
    Trains a recursive feature elimination model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        RFE: Trained recursive feature elimination model
    """
    lr_regressor = LinearRegression(random_state = 42)
    model = RFE(lr_regressor)
    return model

def train_lasso(X_train, y_train):
    """
    Trains a Lasso regression model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        Lasso: Best trained Lasso model after hyperparameter tuning
    """
    lasso = Lasso(random_state = 42)
    scoring_method = 'neg_mean_squared_error'
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    lasso_regressor = GridSearchCV(lasso, parameters, scoring=scoring_method, cv=3)
    lasso_regressor.fit(X_train, y_train)
    model = lasso_regressor.best_estimator_
    return model

def train_ridge(X_train, y_train):
    """
    Trains a Ridge regression model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        Ridge: Best trained Ridge model after hyperparameter tuning
    """
    ridge = Ridge(random_state = 42)
    scoring_method = 'neg_mean_squared_error'
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    ridge_regressor = GridSearchCV(ridge, parameters, scoring=scoring_method, cv=3)
    ridge_regressor.fit(X_train, y_train)
    model = ridge_regressor.best_estimator_
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        RandomForestRegressor: Best trained Random Forest model after hyperparameter tuning
    """
    random_grid = {
        'max_features': ['sqrt'],
        'min_samples_leaf': [0.05,0.1,0.2],
        'min_samples_split': np.linspace(0.1, 1, 10, endpoint=True),
        'n_estimators': [75,100,200]
    }
    scoring_method = 'neg_mean_squared_error'
    n_models = 1
    for key, val in random_grid.items():
        n_models *= len(val)
    n_jobs_per_model = min(max(1, n_cpus//n_models), n_cpus)
    rf = RandomForestRegressor(random_state=42, n_jobs= n_jobs_per_model)
    randomforest_regressor = GridSearchCV(estimator=rf, 
                                        param_grid=random_grid,
                                        cv=3, 
                                        n_jobs=n_cpus // n_jobs_per_model,
                                        scoring=scoring_method, 
                                        verbose=0)  
    randomforest_regressor.fit(X_train, y_train)
    model = randomforest_regressor.best_estimator_
    return model

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        SVR: Best trained SVM model after hyperparameter tuning
    """
    svr = SVR(kernel = 'rbf')
    param_grid_svm = {'C':[0.001, 0.1, 1],'gamma': [1e-7,0.1]}
    scoring_method = 'neg_mean_squared_error'
    svm_regressor = GridSearchCV(estimator=svr, param_grid=param_grid_svm, cv=3, n_jobs=-1, scoring=scoring_method, verbose=0)
    svm_regressor.fit(X_train, y_train)
    model = svm_regressor.best_estimator_
    return model

def train_lightgbm(X_train, y_train):
    """
    Trains a LightGBM model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        LGBMRegressor: Best trained LightGBM model after hyperparameter tuning
    """
    param_grid_gbm = {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [100, 250, 500,1000]}
    n_models = 1
    for key, val in param_grid_gbm.items():
        n_models *= len(val)
    n_jobs_per_model = min(max(1, n_cpus//n_models), n_cpus)
    lightgbm = LGBMRegressor(random_state = 42, n_jobs=n_jobs_per_model, verbose=-1)
    scoring_method = 'neg_mean_squared_error'
    gbm_regressor = GridSearchCV(estimator=lightgbm, param_grid=param_grid_gbm,
                               cv=3, n_jobs=n_cpus // n_jobs_per_model, scoring=scoring_method, verbose=0)
    gbm_regressor.fit(X_train, y_train)
    model = gbm_regressor.best_estimator_
    return model

def train_xgb(X_train, y_train):
    """
    Trains an XGBoost model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        XGBRegressor: Best trained XGBoost model after hyperparameter tuning
    """
    param_grid_gbm = {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [100, 250, 500,1000]}
    n_models = 1
    for key, val in param_grid_gbm.items():
        n_models *= len(val)
    n_jobs_per_model = min(max(1, n_cpus//n_models), n_cpus)
    xgb = XGBRegressor(random_state = 42, n_jobs=n_jobs_per_model)
    scoring_method = 'neg_mean_squared_error'
    xgb_regressor = GridSearchCV(estimator=xgb, param_grid=param_grid_gbm,
                               cv=3, n_jobs=n_cpus // n_jobs_per_model, scoring=scoring_method, verbose=0)
    xgb_regressor.fit(X_train, y_train)
    model = xgb_regressor.best_estimator_
    return model

def train_ada(X_train, y_train):
    """
    Trains an AdaBoost model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        
    Returns:
        AdaBoostRegressor: Best trained AdaBoost model after hyperparameter tuning
    """
    ada = AdaBoostRegressor()
    param_grid_ada = {'n_estimators': [20, 100],
                      'learning_rate': [0.01, 0.05, 1]}
    scoring_method = 'r2'
    ada_regressor = GridSearchCV(estimator=ada, param_distributions=param_grid_ada,
                               cv=3, n_jobs=-1, scoring=scoring_method, verbose=0)
    ada_regressor.fit(X_train, y_train)
    model = ada_regressor.best_estimator_
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model using various metrics.
    
    Args:
        model: Trained model to evaluate
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        
    Returns:
        float: Mean squared error of the model predictions
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score
    y_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    explained_variance = explained_variance_score(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return mse

def append_return_table(df_predict, unique_datetime, y_trade_return, trade_tic, current_index):
    """
    Appends predicted returns to the prediction table.
    
    Args:
        df_predict (pd.DataFrame): DataFrame to append predictions to
        unique_datetime (list): List of unique dates
        y_trade_return (array): Array of predicted returns
        trade_tic (array): Array of ticker symbols
        current_index (int): Current index in the datetime list
    """
    tmp_table = pd.DataFrame(columns=trade_tic)
    tmp_table = tmp_table.append(pd.Series(y_trade_return, index=trade_tic), ignore_index=True)
    df_predict.loc[unique_datetime[current_index]][tmp_table.columns] = tmp_table.loc[0]

def run_4model(df, features_column, label_column, date_column, tic_column,
              unique_ticker, unique_datetime, trade_date, 
              first_trade_date_index=20,
              testing_windows=4,
              max_rolling_window_index=44):
    """
    Runs multiple models (Random Forest, LightGBM, XGBoost) and selects the best performing one.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the data
        features_column (list): List of feature column names
        label_column (str): Name of the label column
        date_column (str): Name of the date column
        tic_column (str): Name of the ticker column
        unique_ticker (list): List of unique ticker symbols
        unique_datetime (list): List of unique dates
        trade_date (list): List of trading dates
        first_trade_date_index (int): Index of first trade date
        testing_windows (int): Number of testing windows
        max_rolling_window_index (int): Maximum rolling window index
        
    Returns:
        tuple: (df_predict_rf, df_predict_gbm, df_predict_xgb, df_predict_best, 
                df_best_model_name, evaluation_record, df_evaluation)
    """
    df_predict_rf = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_predict_gbm = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_predict_xgb = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_predict_best = pd.DataFrame(columns=unique_ticker, index=trade_date)
    df_best_model_name = pd.DataFrame(columns=['model_name'], index=trade_date)
    evaluation_record = {}
    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    for i in range(first_trade_date_index, len(unique_datetime)):
        try:
            if testing_windows >= i:
                continue
            X_train, y_train = prepare_rolling_train(df, 
                                                 features_column,
                                                 label_column,
                                                 date_column, 
                                                 unique_datetime, 
                                                 testing_windows, 
                                                 first_trade_date_index, 
                                                 max_rolling_window_index,
                                                 current_index=i
                                                 )
            X_test, y_test = prepare_rolling_test(df, 
                                              features_column,
                                              label_column,
                                              date_column, 
                                              unique_datetime, 
                                              testing_windows, 
                                              first_trade_date_index,
                                              current_index=i)
            X_trade, y_trade, trade_tic = prepare_trade_data(df,
                                                         features_column,
                                                         label_column,
                                                         date_column,
                                                         tic_column, 
                                                         unique_datetime, 
                                                         testing_windows, 
                                                         first_trade_date_index, 
                                                         current_index=i)
            t = time.perf_counter()
            xgb_model = train_xgb(X_train, y_train)
            print(f"xgb:{time.perf_counter() - t}s")
            t = time.perf_counter()
            gbm_model = train_lightgbm(X_train, y_train)
            print(f"gbm:{time.perf_counter() - t}s")
            t =time.perf_counter()
            rf_model = train_random_forest(X_train, y_train)
            print(f"rf:{time.perf_counter() - t}s")
            rf_eval = evaluate_model(rf_model, X_test, y_test)
            xgb_eval = evaluate_model(xgb_model, X_test, y_test)
            gbm_eval = evaluate_model(gbm_model, X_test ,y_test)
            y_trade_rf = rf_model.predict(X_trade)
            y_trade_xgb = xgb_model.predict(X_trade)
            y_trade_gbm = gbm_model.predict(X_trade)
            eval_data = [
                     [rf_eval, y_trade_rf] ,
                     [xgb_eval, y_trade_xgb],
                     [gbm_eval, y_trade_gbm]
                            ]
            eval_table = pd.DataFrame(eval_data, columns=['model_eval', 'model_predict_return'],
                                          index=['rf', 'xgb', 'gbm'])        
            evaluation_record[unique_datetime[i]]=eval_table
            print(evaluation_record, "Evaluation Record", i)
            y_trade_best = eval_table.model_predict_return.values[eval_table.model_eval == eval_table.model_eval.min()][0]
            best_model_name = eval_table.index.values[eval_table.model_eval == eval_table.model_eval.min()][0]
            df_best_model_name.loc[unique_datetime[i]] = best_model_name
            append_return_table(df_predict_rf, unique_datetime, y_trade_rf, trade_tic, current_index=i)
            append_return_table(df_predict_xgb, unique_datetime, y_trade_xgb, trade_tic, current_index=i)
            append_return_table(df_predict_gbm, unique_datetime, y_trade_gbm, trade_tic, current_index=i)
            append_return_table(df_predict_best, unique_datetime, y_trade_best, trade_tic, current_index=i)
            print('Trade Date: ', unique_datetime[i])
        except Exception:
            traceback.print_exc()
    df_evaluation = get_model_evaluation_table(evaluation_record,trade_date)
    return (
            df_predict_rf, 
            df_predict_gbm,
            df_predict_xgb,
            df_predict_best,
            df_best_model_name, 
            evaluation_record,
            df_evaluation)

def get_model_evaluation_table(evaluation_record, trade_date):
    """
    Creates a table of model evaluation metrics.
    
    Args:
        evaluation_record (dict): Dictionary containing evaluation records
        trade_date (list): List of trading dates
        
    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each model
    """
    evaluation_list = []
    for d in trade_date:
        try:
            evaluation_list.append(evaluation_record[d]['model_eval'].values)
        except:
            print('error')
    df_evaluation = pd.DataFrame(evaluation_list,columns = ['rf', 'xgb', 'gbm'])
    print(df_evaluation)
    df_evaluation.index = trade_date
    return df_evaluation

def save_model_result(sector_result, sector_name):
    """
    Saves model results to CSV files.
    
    Args:
        sector_result (tuple): Tuple containing model results
        sector_name (str): Name of the sector
    """
    df_predict_rf = sector_result[0].astype(np.float64)
    df_predict_gbm = sector_result[1].astype(np.float64)
    df_predict_xgb = sector_result[2].astype(np.float64)
    df_predict_best = sector_result[3].astype(np.float64)

    df_best_model_name = sector_result[4]
    df_evaluation_score = sector_result[5]
    df_model_score = sector_result[6]

    filename = 'results/'+sector_name+'/'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    df_predict_rf.to_csv('results/'+sector_name+'/df_predict_rf.csv')
    df_predict_gbm.to_csv('results/'+sector_name+'/df_predict_gbm.csv')
    df_predict_xgb.to_csv('results/'+sector_name+'/df_predict_xgb.csv')
    df_predict_best.to_csv('results/'+sector_name+'/df_predict_best.csv')
    df_best_model_name.to_csv('results/'+sector_name+'/df_best_model_name.csv')
    #df_evaluation_score.to_csv('results/'+sector_name+'/df_evaluation_score.csv')
    df_model_score.to_csv('results/'+sector_name+'/df_model_score.csv')

def calculate_sector_daily_return(daily_price, unique_ticker, trade_date):
    """
    Calculates daily returns for stocks in a sector.
    
    Args:
        daily_price (pd.DataFrame): DataFrame containing daily prices
        unique_ticker (list): List of unique ticker symbols
        trade_date (list): List of trading dates
        
    Returns:
        pd.DataFrame: DataFrame containing daily returns
    """
    daily_price_pivot = pd.pivot_table(daily_price, values='adj_price', index=['datadate'],
                   columns=['tic'], aggfunc=np.mean)
    daily_price_pivot=daily_price_pivot[unique_ticker]
    daily_return=daily_price_pivot.pct_change()
    daily_return = daily_return[daily_return.index>=trade_date[0]]
    return daily_return

def calculate_sector_quarterly_return(daily_price, unique_ticker, trade_date_plus1):
    """
    Calculates quarterly returns for stocks in a sector.
    
    Args:
        daily_price (pd.DataFrame): DataFrame containing daily prices
        unique_ticker (list): List of unique ticker symbols
        trade_date_plus1 (list): List of trading dates plus one period
        
    Returns:
        pd.DataFrame: DataFrame containing quarterly returns
    """
    daily_price_pivot = pd.pivot_table(daily_price, values='adj_price', index=['datadate'],
                   columns=['tic'], aggfunc=np.mean)
    daily_price_pivot=daily_price_pivot[unique_ticker]
    quarterly_price_pivot=daily_price_pivot.ix[trade_date_plus1]
    quarterly_return=quarterly_price_pivot.pct_change()
    quarterly_return = quarterly_return[quarterly_return.index>trade_date_plus1[0]]
    return quarterly_return

def pick_stocks_based_on_quantiles(df_predict_best):
    """
    Picks stocks based on predicted return quantiles.
    
    Args:
        df_predict_best (pd.DataFrame): DataFrame containing best predictions
        
    Returns:
        tuple: (quantile_0_30, quantile_70_100) containing stocks in bottom 30% and top 30% quantiles
    """
    quantile_0_30 = {}
    quantile_70_100 = {}
    for i in range(df_predict_best.shape[0]):
        q_30=df_predict_best.iloc[i].quantile(0.3)
        q_70=df_predict_best.iloc[i].quantile(0.7)
        quantile_0_30[df_predict_best.index[i]] = df_predict_best.iloc[i][df_predict_best.iloc[i] <= q_30]
        quantile_70_100[df_predict_best.index[i]] = df_predict_best.iloc[i][(df_predict_best.iloc[i] >= q_70)]
    return (quantile_0_30, quantile_70_100)

def calculate_portfolio_return(daily_return, trade_date_plus1, long_dict, frequency_date):
    """
    Calculates portfolio returns based on daily returns.
    
    Args:
        daily_return (pd.DataFrame): DataFrame containing daily returns
        trade_date_plus1 (list): List of trading dates plus one period
        long_dict (dict): Dictionary containing long positions
        frequency_date (list): List of frequency dates
        
    Returns:
        pd.DataFrame: DataFrame containing portfolio returns
    """
    df_portfolio_return = pd.DataFrame(columns=['portfolio_return'])
    for i in range(len(trade_date_plus1) - 1):
        long_tic_return_daily = \
            daily_return[(daily_return.index >= trade_date_plus1[i]) &\
                         (daily_return.index < trade_date_plus1[i + 1])][long_dict[trade_date_plus1[i]].index]
        long_daily_return = long_tic_return_daily 
        df_temp = long_daily_return.mean(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['daily_return'])
        df_portfolio_return = df_portfolio_return.append(df_temp)
    return df_portfolio_return

def calculate_portfolio_quarterly_return(quarterly_return, trade_date_plus1, long_dict):
    """
    Calculates portfolio returns based on quarterly returns.
    
    Args:
        quarterly_return (pd.DataFrame): DataFrame containing quarterly returns
        trade_date_plus1 (list): List of trading dates plus one period
        long_dict (dict): Dictionary containing long positions
        
    Returns:
        pd.DataFrame: DataFrame containing portfolio quarterly returns
    """
    df_portfolio_return = pd.DataFrame(columns=['portfolio_return'])
    for i in range(len(trade_date_plus1) - 1):
        long_tic_return = quarterly_return[quarterly_return.index == trade_date_plus1[i + 1]][long_dict[trade_date_plus1[i]].index]
        df_temp = long_tic_return.mean(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['portfolio_return'])
        df_portfolio_return = df_portfolio_return.append(df_temp)
    return df_portfolio_return

def long_only_strategy_daily(df_predict_return, daily_return, trade_month_plus1, top_quantile_threshold=0.75):
    """
    Implements a long-only trading strategy using daily returns.
    
    Args:
        df_predict_return (pd.DataFrame): DataFrame containing predicted returns
        daily_return (pd.DataFrame): DataFrame containing daily returns
        trade_month_plus1 (list): List of trading months plus one period
        top_quantile_threshold (float): Threshold for selecting top performing stocks
        
    Returns:
        pd.DataFrame: DataFrame containing daily portfolio returns
    """
    long_dict = {}
    for i in range(df_predict_return.shape[0]):
        top_q = df_predict_return.iloc[i].quantile(top_quantile_threshold)
        long_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][df_predict_return.iloc[i] >= top_q]
    df_portfolio_return_daily = pd.DataFrame(columns=['daily_return'])
    for i in range(len(trade_month_plus1) - 1):
        long_normalize_weight = 1/long_dict[trade_month_plus1[i]].shape[0]
        long_tic_return_daily = \
        daily_return[(daily_return.index >= trade_month_plus1[i]) & (daily_return.index < trade_month_plus1[i + 1])][
            long_dict[trade_month_plus1[i]].index]
        long_daily_return = long_tic_return_daily * long_normalize_weight
        df_temp = long_daily_return.sum(axis=1)
        df_temp = pd.DataFrame(df_temp, columns=['daily_return'])
        df_portfolio_return_daily = df_portfolio_return_daily.append(df_temp)
    return df_portfolio_return_daily

def long_only_strategy_monthly(df_predict_return, tic_monthly_return, trade_month, top_quantile_threshold=0.7):
    """
    Implements a long-only trading strategy using monthly returns.
    
    Args:
        df_predict_return (pd.DataFrame): DataFrame containing predicted returns
        tic_monthly_return (pd.DataFrame): DataFrame containing monthly returns
        trade_month (list): List of trading months
        top_quantile_threshold (float): Threshold for selecting top performing stocks
        
    Returns:
        pd.Series: Series containing monthly portfolio returns
    """
    long_dict = {}
    for i in range(df_predict_return.shape[0]):
        top_q = df_predict_return.iloc[i].quantile(top_quantile_threshold)
        long_dict[df_predict_return.index[i]] = df_predict_return.iloc[i][df_predict_return.iloc[i] >= top_q]
    portfolio_return_dic = {}
    for i in range(len(trade_month)):
        long_normalize_weight = long_dict[trade_month[i]] / sum(long_dict[trade_month[i]].values)
        long_tic_return = tic_monthly_return[tic_monthly_return.index == trade_month[i]][
            long_dict[trade_month[i]].index]
        long_return_table = long_tic_return * long_normalize_weight
        portfolio_return_dic[trade_month[i]] = long_return_table.values.sum()
    df_portfolio_return = pd.DataFrame.from_dict(portfolio_return_dic, orient='index')
    df_portfolio_return = df_portfolio_return.reset_index()
    df_portfolio_return.columns = ['trade_month', 'monthly_return']
    df_portfolio_return.index = df_portfolio_return.trade_month
    df_portfolio_return = df_portfolio_return['monthly_return']
    return df_portfolio_return

def plot_predict_return_distribution(df_predict_best, sector_name, out_path):
    """
    Plots the distribution of predicted returns.
    
    Args:
        df_predict_best (pd.DataFrame): DataFrame containing best predictions
        sector_name (str): Name of the sector
        out_path (str): Path to save the plots
    """
    import matplotlib.pyplot as plt
    for i in range(df_predict_best.shape[0]):
        fig=plt.figure(figsize=(8,5))
        df_predict_best.iloc[i].hist()
        plt.xlabel("predicted return",size=15)
        plt.ylabel("frequency",size=15)
        plt.title(sector_name+": trade date - "+str(df_predict_best.index[i]),size=15)
    plt.savefig(out_path+str(df_predict_best.index[i])+".png")

def stock_selection():
    """
    Selects stocks based on predicted returns across different sectors.
    Saves the selected stocks to a CSV file.
    """
    sectors = range(10, 65, 5)
    df_dict = {'gvkey':[], 'predicted_return':[], 'trade_date':[]}
    for sector in sectors:
        try:
            df = pd.read_csv(f"results/sector{sector}/df_predict_best.csv", index_col=0)
            for idx in df.index:
                predicted_return = df.loc[idx]
                top_q = predicted_return.quantile(0.75)
                predicted_return = predicted_return[predicted_return >= top_q]
                for gvkey in predicted_return.index:
                    df_dict["gvkey"].append(gvkey)
                    df_dict["predicted_return"].append(predicted_return[gvkey])
                    df_dict["trade_date"].append(idx)
        except FileNotFoundError:
            continue
    df_result = pd.DataFrame(df_dict)
    df_result.to_csv("results/stock_selected.csv")








