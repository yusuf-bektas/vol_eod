import argparse
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import socket
import time
import warnings
from matplotlib import pyplot as plt


def recvall(sock):
    time.sleep(1)
    BUFF_SIZE = 0x100
    data = bytearray()
    while True:
        packet = sock.recv(BUFF_SIZE)
        data.extend(packet)
        if "GONDERIM SONU" in str(data[-19:-1]):
            break
    return data

def get_historic_data(host, port, asset, start_date, finish_date, freq):
    data = []
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setblocking(1)
    client_socket.connect((host, port))
    message_sent = f"UserName:MATRIKS;Password:MTX;Tip:11;{freq};{asset};{start_date};{finish_date};0;\r\n"
    client_socket.sendall(message_sent.encode())
    message_received = recvall(client_socket)
    messages = str(message_received).replace("bytearray(b''", "").split("\\r\\n")
    for msg in messages:
        data.append(msg.split(";"))
    df = pd.DataFrame(data).iloc[:, 1:]
    df.index = pd.to_datetime(df[1], format='%Y-%m-%d %H:%M:%S')
    df = df.drop([1], axis=1)
    df.columns = ["open", "high", "low", "close", "volume", "vwap"]
    df.index.names = ['time']
    return df.astype(float)

def save_data(df, asset, freq, output_folder):
    output_path = os.path.join(output_folder, f"{asset}_{freq}.csv")
    print(f"Saving data to {output_path}",flush=True)
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(output_path)

def plot_data(df, asset, plot_folder):
    plt.figure(figsize=(14, 7))
    plt.plot(df['close'].reset_index(drop=True), label='Close', color='blue')
    plt.title(f'Data for {asset}')
    plt.xlabel('Index')
    plt.ylabel('Close Values')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, f"{asset}_{datetime.now().strftime('%Y%m%d')}.png")
    print(f"Saving plot to {plot_path}")
    plt.savefig(plot_path)
    plt.close()

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Historic Data Fetcher")
    parser.add_argument("--asset", type=str, required=True, help="Asset name to fetch data for")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    print(f"Fetching data for {args.asset}", flush=True)
    asset = args.asset
    config= json.loads(args.config)
    data_path = config["data_folder"]
    data_path = os.path.join(BASE_DIR, data_path)
    data_check_plot_path = config["data_check_plot_folder"]
    data_check_plot_path = os.path.join(BASE_DIR, data_check_plot_path)
    print(f"Data will be saved to {data_path}", flush=True)

    historic_config = config["historic"]

    host = historic_config["host"]
    port = historic_config["port"]
    start_date = historic_config["start_date"]
    finish_date = datetime.now().strftime('%Y%m%d')
    freq = str(historic_config["freq"])

    try:
        df = get_historic_data(host, port, asset, start_date, finish_date, freq)
        print(f"{asset} data between {start_date} and {finish_date} is received")
        save_data(df, asset, freq, data_path)
        plot_data(df, asset, data_check_plot_path)
    except Exception as e:
        print(f"Error in getting data for {asset} between {start_date} and {finish_date}, closing the program")
        print(e)
        raise e

if __name__ == "__main__": 
    warnings.filterwarnings("ignore")
    print("Starting fetch_hist_data.py...",flush=True)
    main()
