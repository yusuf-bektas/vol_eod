import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

plot_dir = "data_check_plots"

def read_historic_data(path,asset,freq,start_t="10:00:00", end_t="18:00:00",days_to_remove_count=3):
    # Create the directory if it doesn't exist
    
    os.makedirs(plot_dir, exist_ok=True)
    file_path = os.path.join(path, f"{asset}_{freq}.csv")
    eurex = pd.read_csv(file_path, index_col=0, parse_dates=True).dropna()
    
    # Calculate start and end times for each day
    start_times = pd.Series(eurex.index, index=eurex.index).groupby(eurex.index.normalize()).first()
    end_times = pd.Series(eurex.index, index=eurex.index).groupby(eurex.index.normalize()).last()
    
    # Plot the start times
    start_times.apply(lambda x: x.hour * 60 + x.minute).plot(title='Start Times')
    plt.savefig(os.path.join(plot_dir, 'start_times.png'))
    plt.clf()
    
    # Filter data between specified times
    eurex = eurex.between_time(start_t, end_t)
    
    # Plot the closing prices
    eurex.close.reset_index(drop=True).plot(title='Closing Prices')
    plt.savefig(os.path.join(plot_dir, 'closing_prices.png'))
    plt.clf()
    
    # Recalculate start and end times after filtering
    start_times = pd.Series(eurex.index, index=eurex.index).groupby(eurex.index.normalize()).first()
    start_times = start_times.apply(lambda x: x.hour * 60 + x.minute)
    end_times = pd.Series(eurex.index, index=eurex.index).groupby(eurex.index.normalize()).last()
    end_times = end_times.apply(lambda x: x.hour * 60 + x.minute)
    
    # Identify days to remove based on the start and end times
    days_to_remove = start_times.sort_values(ascending=False).iloc[:days_to_remove_count].index
    days_to_remove = days_to_remove.union(end_times.sort_values(ascending=False).iloc[:days_to_remove_count].index)
    
    # Filter out the identified days
    eurex = eurex[~eurex.index.normalize().isin(days_to_remove)]
    
    # Recalculate start and end times after removing specific days
    start_times = pd.Series(eurex.index, index=eurex.index).groupby(eurex.index.normalize()).first()
    start_times = start_times.apply(lambda x: x.hour * 60 + x.minute)
    end_times = pd.Series(eurex.index, index=eurex.index).groupby(eurex.index.normalize()).last()
    end_times = end_times.apply(lambda x: x.hour * 60 + x.minute)
    
    # Plot the final start and end times
    start_times.plot(title='Filtered Start Times')
    plt.savefig(os.path.join(plot_dir, 'filtered_start_times.png'))
    plt.clf()
    end_times.plot(title='Filtered End Times')
    plt.savefig(os.path.join(plot_dir, 'filtered_end_times.png'))
    plt.clf()

    return eurex

def gk_vol(eurex,period="5T"):
    eurex['vol']=0.5*(np.log(eurex.high/eurex.low)**2)-((2*np.log(2)-1)*(np.log(eurex.close/eurex.open)**2))
    eurex['vol']=eurex['vol'].rolling(period).mean()
    
    eurex['vol'].groupby(eurex.index.time).median().plot()
    plt.title("Volatility")
    plt.savefig(os.path.join(plot_dir, 'median_volatility_by_time.png'))
    plt.clf()
    return eurex

def calc_seasonal_vol(eurex,seasonal_period=20):
    vol_ssnl=eurex.vol.groupby([eurex.index.normalize(),eurex.index.time]).last().unstack()
    vol_ssnl=vol_ssnl.rolling(window=seasonal_period,closed='left').median().stack(dropna=False)
    new_index=vol_ssnl.index.get_level_values(0).normalize().astype(str)+' '+vol_ssnl.index.get_level_values(1).astype(str)
    new_index=pd.to_datetime(new_index)
    vol_ssnl.index=new_index
    eurex['vol_ssnl']=vol_ssnl
    eurex=eurex.dropna()
    eurex.vol_ssnl.groupby(eurex.index.time).median().plot()
    eurex.vol.groupby(eurex.index.time).median().plot()
    plt.title("Median Volatility vs Seasonal Volatility by Time")
    plt.savefig(os.path.join(plot_dir, 'vol_vs_seasonal_vol_by_time.png'))
    plt.clf()
    return eurex

def create_daily_preds(eurex,daily_ewm_span=2):
    eurex_daily=eurex.resample('D').agg({'open':'first','high':'max','low':'min','close':'last','vol':'sum','vol_ssnl':'sum'}).dropna()
    eurex_daily.vol_ssnl.plot()
    eurex_daily.vol.plot()

    eurex_daily['vol_sum_pred']=eurex_daily.vol.ewm(daily_ewm_span).mean()
    eurex_daily['vol_sum_pred']=eurex_daily['vol_sum_pred'].shift(1)
    eurex_daily.vol_sum_pred.plot()
    eurex_daily.vol.plot()
    plt.title("Daily Volatility")
    plt.savefig(os.path.join(plot_dir, 'daily_vol_vs_seasonal_vol.png'))
    plt.clf()
    return eurex_daily

def calc_daily_adjustment(eurex,eurex_daily):
    eurex['vol_sum_pred']=eurex.index.normalize().map(eurex_daily.vol_sum_pred)

    eurex['vol_ssnl_sum']=eurex.index.normalize().map(eurex.vol_ssnl.groupby(eurex.index.normalize()).sum())
    eurex['daily_pred_adj_r']=eurex.vol_sum_pred/eurex.vol_ssnl_sum
    eurex['daily_pred_adj_r'].plot()
    plt.title("Daily Prediction Adjustment Ratio")
    plt.savefig(os.path.join(plot_dir, 'daily_pred_adj_ratio.png'))
    plt.clf()
    return eurex

def adjust_seasonal_to_daily(eurex):
    eurex['vol_ssnl_adj']=eurex.vol_ssnl*eurex.daily_pred_adj_r
    eurex['count_to_eod']=eurex.index.normalize().map(eurex.vol.groupby(eurex.index.normalize()).count())-eurex.vol.groupby(eurex.index.normalize()).cumcount()
    eurex['int_vol_eod']=eurex.index.normalize().map(eurex.vol.groupby(eurex.index.normalize()).sum())-eurex.vol.groupby(eurex.index.normalize()).cumsum()
    eurex['mean_vol_eod']=eurex.int_vol_eod/eurex.count_to_eod

    eurex['int_vol_eod_ssnl_adj']=eurex.index.normalize().map(eurex.vol_ssnl_adj.groupby(eurex.index.normalize()).sum())-eurex.vol_ssnl_adj.groupby(eurex.index.normalize()).cumsum()
    eurex['mean_vol_eod_ssnl_adj']=eurex.int_vol_eod_ssnl_adj/eurex.count_to_eod

    eurex.mean_vol_eod.groupby(eurex.index.time).mean().plot()
    eurex.mean_vol_eod_ssnl_adj.groupby(eurex.index.time).mean().plot()
    plt.title("Mean Volatility vs Seasonal Adjusted Mean Volatility by Time")
    plt.savefig(os.path.join(plot_dir, 'mean_vol_vs_seasonal_adj_mean_vol_by_time.png'))
    plt.clf()
    return eurex

def fit_seasonal_curve(eurex,spline_s=1, spline_k=5):
    from scipy.interpolate import UnivariateSpline

    vol_ssnl = eurex.vol_ssnl_adj.between_time("10:20", "18:00")
    vol_ssnl = pd.DataFrame(vol_ssnl).rename(columns={'vol_ssnl_adj': 'vol_ssnl'})
    vol_ssnl['time_p'] = vol_ssnl.vol_ssnl.groupby(vol_ssnl.index.normalize()).cumcount() / vol_ssnl.index.normalize().map(vol_ssnl.vol_ssnl.groupby(vol_ssnl.index.normalize()).count())

    daily_vol_ssnl_fits = []
    last_day_spline = None
    splines_by_day = {}


    for day in vol_ssnl.index.normalize().unique():
        day_data = vol_ssnl.loc[vol_ssnl.index.normalize() == day]
        day_data = day_data.dropna()
        if day_data.shape[0] > 0:
            y = day_data.vol_ssnl
            x = day_data.time_p
            spline = UnivariateSpline(x, y, s=spline_s, k=spline_k)
            yhats = spline(x)
            daily_vol_ssnl_fits.append(pd.Series(yhats, index=day_data.index))
            last_day_spline = spline
            splines_by_day[day] = spline


    daily_vol_ssnl_fits = pd.concat(daily_vol_ssnl_fits)
    vol_ssnl['vol_ssnl_fit'] = daily_vol_ssnl_fits
    eurex['vol_ssnl_fit'] = vol_ssnl['vol_ssnl_fit']
    eurex['time_p']=eurex.groupby(eurex.index.normalize()).vol.cumcount()/eurex.index.normalize().map(eurex.groupby(eurex.index.normalize()).vol.count())
    eurex=eurex.dropna()
    
    def average_spline_value(model, t):
        from scipy.integrate import quad

        if not 0 <= t <= 1:
            raise ValueError("t must be between 0 and 1")

        # Integrate the spline from t to 1
        integral, _ = quad(model, t, 1)
        
        # Calculate the average value
        average_value = integral / (1 - t)
        
        return average_value
    
    splines_by_day=pd.DataFrame(splines_by_day,index=['spline']).T

    eurex['spline_obj']=eurex.index.normalize().map(splines_by_day.spline)
    
    eurex['mean_vol_eod_ssnl_adj']=eurex.apply(lambda x: average_spline_value(x.spline_obj,x.time_p),axis=1)

    # Plotting the average by time
    vol_ssnl.vol_ssnl.groupby(vol_ssnl.index.time).median().plot()
    vol_ssnl.vol_ssnl_fit.groupby(vol_ssnl.index.time).median().plot()
    plt.title("Seasonal Volatility vs Seasonal Volatility Fit by Time")
    plt.savefig(os.path.join(plot_dir, 'seasonal_vol_vs_seasonal_vol_fit_by_time.png'))
    plt.clf()

    return eurex, last_day_spline

def fit_model(eurex,start_time="10:20:00",end_time="18:00:00", span_vals=[2,20],outlier_qtl=1):
    eurex['vol_diff']=np.log(eurex.vol).diff()
    span_vals=[2,20]
    X=pd.DataFrame()
    for span in span_vals:
        eurex['vol_diff_ewm_'+str(span)]=eurex.vol_diff.ewm(span=span).mean()
        eurex['vol_ewm_'+str(span)]=eurex.vol.ewm(span=span).mean()
        X['vol_diff_ewm_'+str(span)]=eurex['vol_diff_ewm_'+str(span)]
        
    for span in span_vals:
        eurex[f'vol_ewm_{span}_ssnl_adj_diff']=np.log(eurex[f'vol_ewm_{span}']/eurex.vol_ssnl_fit)
        X[f'vol_ewm_{span}_ssnl_adj_diff']=eurex[f'vol_ewm_{span}_ssnl_adj_diff']
        
    X=X.shift(1).dropna()
    y=np.log(eurex.mean_vol_eod/eurex.mean_vol_eod_ssnl_adj).replace([np.inf,-np.inf],np.nan).dropna()
    #X=eurex[['vol_ewm_2_ssnl_adj_diff','vol_ewm_10_ssnl_adj_diff','vol_diff_ewm_10','vol_diff_ewm_2']].dropna()
    #outlier elimination
    if outlier_qtl<1:
        qtl=outlier_qtl
        y=y[np.abs(y)<np.abs(y).quantile(qtl)]

    X=X.between_time(start_time,end_time)

    common_index=X.index.intersection(y.index)

    X=X.loc[common_index]
    y=y.loc[common_index]
    import statsmodels.api as sm
    ols_model=sm.OLS(y,X).fit()
    print(ols_model.summary())
    return ols_model

def save_models(model_params, last_day_spline, asset, freq, spline_s, spline_k, final_model_output_folder=None, seasonal_model_output_folder=None):
    # model_output_folder = "/home/qrdutil/peoples_workspaces/yusuf-workspace/eurex/tt-stream-server-connector/run/model/ols"
    # spline_output_folder = "/home/qrdutil/peoples_workspaces/yusuf-workspace/eurex/tt-stream-server-connector/run/model/seasonal_spline"
    if seasonal_model_output_folder is None or final_model_output_folder is None:
        raise ValueError("Please provide a valid output folder for models")
    
    final_model_output_path=os.path.join(os.getcwd(),final_model_output_folder)
    seasonal_model_output_path=os.path.join(os.getcwd(),seasonal_model_output_folder)
    os.makedirs(final_model_output_folder, exist_ok=True)
    os.makedirs(seasonal_model_output_folder, exist_ok=True)

    model_filename = os.path.join(final_model_output_path, f"{asset}_{freq}.json")
    seasonal_model_filename = os.path.join(seasonal_model_output_path, f"{asset}_{freq}_{spline_s}_{spline_k}.pkl")
    import json
    with open(model_filename, 'w') as f:
        json.dump(model_params, f)
    print(f"Final Model parameters saved to {model_filename}")

    import pickle
    if last_day_spline is not None:
        # Saving spline
        with open(seasonal_model_filename, 'wb') as f:
            pickle.dump(last_day_spline, f)
        print(f"Seasonal model saved to {seasonal_model_filename}")
    else:
        print("Error: last_day_spline is None. Spline was not saved.")

    
def main():
    import argparse
    import json
    #reading conf file
    #conf_path="/home/qrdutil/peoples_workspaces/yusuf-workspace/eurex/tt-stream-server-connector/run/model/model_config.json"

    parser = argparse.ArgumentParser(description='Run the Eurex TT Stream Server Connector model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--asset', type=str, required=True, help='Asset name.')
    parser.add_argument('--freq', type=str, required=True, help='Frequency.')

    args = parser.parse_args()
    asset = args.asset
    freq = args.freq
    config=json.loads(args.config)
    data_path = config['data_folder']

    data_path=os.path.join(os.getcwd(),data_path)
    #asset="FSX5E"
    #freq="1"
    eurex=read_historic_data(data_path,asset=asset,freq=freq)
    vol_period=config['intraday']['vol_window']
    eurex=gk_vol(eurex,period=vol_period)


    eurex=calc_seasonal_vol(eurex,seasonal_period=config['historic']['seasonality_period'])
    eurex_daily=create_daily_preds(eurex,daily_ewm_span=config['historic']['ewm_span'])
    eurex=calc_daily_adjustment(eurex,eurex_daily)
    eurex=adjust_seasonal_to_daily(eurex)
    spline_s=config['historic']['spline_s']
    spline_k=config['historic']['spline_k']
    eurex,last_day_spline=fit_seasonal_curve(eurex, spline_s=spline_s, spline_k=spline_k)


    start_time=config['intraday']['start_time']
    end_time=config['intraday']['end_time']
    span_vals=config['intraday']['ewm_span']
    outlier_qtl=config['intraday']['outlier_qtl']
    ols_model=fit_model(eurex,start_time=start_time,end_time=end_time, span_vals=span_vals,outlier_qtl=outlier_qtl)

    model_params = ols_model.params.to_dict()

    final_model_output_folder = config['final_model_output_folder']
    seasonal_model_output_folder = config['seasonal_model_output_folder']
    save_models(model_params, last_day_spline, asset, freq, spline_s, spline_k, final_model_output_folder, seasonal_model_output_folder)
    

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    main()
