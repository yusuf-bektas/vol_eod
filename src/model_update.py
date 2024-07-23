import redis
import json
import pandas as pd
import numpy as np
import time
import os
import warnings
import redis



class VolEod:
    def __init__(self, assetTT,redis_client, window=20, period="1T",vol_window="5T",ewm_spans=[2,20],
                 start_time="10:20",end_time="19:00", seasonality_path=None, daily_adj_r=1):
        self.assetTT = assetTT
        self.window = window+1
        self.redis_client=redis_client
        self.period = period
        self.vol_window = vol_window
        self.ohlc_data = pd.DataFrame()
        self.vols = pd.Series()
        self.ewm_spans=ewm_spans
        self.start_time=pd.Timestamp(start_time)
        self.end_time=pd.Timestamp(end_time)
        self.time_p=self.calc_time_p()
        self.X=pd.DataFrame()
        

    def load_seasonal_model(self,path):
        import pickle
        with open(path, 'rb') as f:
            self.seasonal_model = pickle.load(f)

    def load_ols_params(self,path):
        self.ols_params=json.load(open(path))


    def calc_time_p(self):
        """
        calculates what is the time portion now w.r.t start and end time.
        returns a float between 0 and 1
        """
        return (pd.Timestamp.now()-self.start_time)/(self.end_time-self.start_time)
    

    def average_spline_value(model, t):
        from scipy.integrate import quad

        if not 0 <= t <= 1:
            raise ValueError("t must be between 0 and 1")

        # Integrate the spline from t to 1
        integral, _ = quad(model, t, 1)
        
        # Calculate the average value
        average_value = integral / (1 - t)
        
        return average_value

    def resample_to_ohlc(self, trades):
        if not trades:
            return pd.DataFrame()

        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        # Resample to 1-minute OHLC
        ohlc = df['price'].resample(self.period).ohlc()
        ohlc['volume'] = df['qty'].resample(self.period).sum()
        return ohlc

    def update_ohlc_data(self):
        recent_trades = self.get_recent_trades()
        self.ohlc_data = self.resample_to_ohlc(recent_trades)

    def get_recent_trades(self):
        assetTT_key = f"{self.assetTT.replace(' ', '_')}_trades"
        current_time = time.time()
        cutoff_time = current_time - self.window * 60
        trades = self.redis_client.zrangebyscore(assetTT_key, cutoff_time, current_time)
        return [json.loads(trade) for trade in trades]
    
    def calc_vol(self):
        self.vols =0.5*np.log(self.ohlc_data['high']/self.ohlc_data['low'])**2 - (2*np.log(2)-1)*(np.log(self.ohlc_data['close']/self.ohlc_data['open']))**2
        self.vols = self.vols.rolling(window=self.vol_window).mean()
        #print(self.vols)
    
    def calc_seasonal_features(self):
        seasonal_pred=self.seasonal_model(self.time_p)
        X=self.X
        for span in self.ewm_spans:
            vol_ewm=self.vols.ewm(span=span).mean()
            X[f'vol_ewm_{span}_ssnl_adj_diff']=np.log(vol_ewm)-np.log(seasonal_pred)

    def calc_intr_features(self):
        vol_diff=np.log(self.vols).diff().dropna()
        X=self.X
        for span in self.ewm_spans:
            X['vol_diff_ewm_'+str(span)]=vol_diff.ewm(span=span).mean()

    
    def get_pred(self):
        X=self.X.iloc[-1]
        params=self.ols_params
        res=0
        for key in params:
            res+=params[key]*X[key]
        average_spline_pred=self.average_spline_value(self.time_p)
        mean_vol_eod=average_spline_pred*np.exp(res)
        #int_vol_eod=mean_vol_eod*(1-self.time_p)
        return np.sqrt(mean_vol_eod*480*252)
        

    def average_spline_value(self, t):
        from scipy.integrate import quad

        if not 0 <= t <= 1:
            raise ValueError("t must be between 0 and 1")

        # Integrate the spline from t to 1
        integral, _ = quad(self.seasonal_model, t, 1)
        
        # Calculate the average value
        average_value = integral / (1 - t)
        
        return average_value

        
    def update(self):
        self.update_ohlc_data()
        self.calc_vol()     
        self.calc_intr_features()
        self.calc_seasonal_features()
        
        print(f"{self.get_pred():.3f}")


def get_assetMSS(assetTT):
    if assetTT == 'FESX Sep24':
        return 'FSX5E'
    else :
        raise ValueError(f"Unknown assetTT: {assetTT}")

def main(assetTT,assetMSS,cnf_main,cnf_model):
    #cnf_path="/home/qrdutil/peoples_workspaces/yusuf-workspace/eurex/tt-stream-server-connector/run/model/gen_config.json"
    # import argparse
    # print("Starting model update script...",flush=True)
    
    # parser = argparse.ArgumentParser(description="Live model update")
    # parser.add_argument("--assetTT", type=str, required=True, help="Asset name to subscribe to (TT format)")
    # parser.add_argument("--assetMSS", type=str, required=True, help="Asset name to subscribe to (MSS format)")
    # parser.add_argument("--configMain", type=str, required=True, help="config as json")
    # parser.add_argument("--configModel", type=str, required=True, help="config as json")
    
    # args = parser.parse_args()
    # print(f"Getting data for {args.assetTT} from redis", flush=True)
    # assetTT = args.assetTT.strip("'")
    # assetMSS = args.assetMSS.strip("'")
    # cnf_main = json.loads(args.configMain)
    # cnf_model = json.loads(args.configModel)

    warnings.filterwarnings("ignore")

    host=cnf_main['redis']['host']
    port=cnf_main['redis']['port']
    db=cnf_main['redis']['db']
    
    redis_client = redis.Redis(host=host, port=port, db=db)
    ############################
    book_channel = cnf_main['redis']['book_channel']
    ############################
    trade_channel=cnf_main['redis']['trade_channel']
    pubsub = redis_client.pubsub()
    #pubsub.subscribe(book_channel)
    pubsub.subscribe(trade_channel)
    print(f"Subscribed to {trade_channel}. Waiting for messages...",flush=True)
    #print(f"Subscribed to {book_channel}. Waiting for messages...")

    vol_window=cnf_model['intraday']['vol_window']
    model = VolEod(assetTT,redis_client,vol_window=vol_window)

    seasonal_output_folder=cnf_model['seasonal_model_output_folder']
    

    asset=assetMSS
    freq=cnf_model['historic']['freq']
    spline_s=cnf_model['historic']['spline_s']
    spline_k=cnf_model['historic']['spline_k']
    spline_filename = os.path.join(seasonal_output_folder, f"{asset}_{freq}_{spline_s}_{spline_k}.pkl")
    model.load_seasonal_model(spline_filename)

    #ols_path="/home/qrdutil/peoples_workspaces/yusuf-workspace/eurex/tt-stream-server-connector/run/model/ols"
    final_model_path=cnf_model['final_model_output_folder']
    ols_filename = os.path.join(final_model_path, f"{asset}_{freq}.json")
    model.load_ols_params(ols_filename)
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            #print(f"Received: {message['data'].decode('utf-8')}")
            try:
                model.update()
            except Exception as e:
                print(e)


if __name__ == "__main__":
    print("Starting model update script...",flush=True)
    #warnings.filterwarnings("ignore")
    main()

