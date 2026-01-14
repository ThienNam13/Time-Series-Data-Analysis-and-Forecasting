import pandas as pd


def load_electricity_data(path):
    """
    Load UCI Electricity Load Diagrams (LD2011_2014.txt)
    Output: DataFrame với cột [load]
    """
    df = pd.read_csv(
        path,
        sep=';',
        decimal=',',
        index_col=0
    )

    # index -> datetime
    df.index = pd.to_datetime(df.index)

    # Lấy trung bình điện năng tiêu thụ của tất cả customers
    df['load'] = df.mean(axis=1)

    # Chỉ giữ lại biến load
    df = df[['load']]

    return df


def load_weather_data(path):
    # Bỏ 2 dòng metadata
    df = pd.read_csv(path, skiprows=2)

    # Đổi tên cột time
    df = df.rename(columns={'time': 'timestamp'})

    # Rename bằng keyword (chống lỗi encoding)
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'temperature' in col_lower:
            rename_map[col] = 'temperature'
        elif 'humidity' in col_lower:
            rename_map[col] = 'humidity'
        elif 'wind_speed' in col_lower or 'wind speed' in col_lower:
            rename_map[col] = 'wind_speed'

    df = df.rename(columns=rename_map)

    # Chuyển timestamp & SET INDEX (QUAN TRỌNG NHẤT)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Chỉ giữ biến cần
    df = df[['temperature', 'humidity', 'wind_speed']]

    return df



def resample_data(df, freq='H'):
    """
    Resample dữ liệu về cùng frequency
    freq: 'H' (hourly) hoặc 'D' (daily)
    """
    return df.resample(freq).mean()


def merge_load_weather(load_df, weather_df):
    """
    Merge load & weather theo timestamp
    """
    df = load_df.merge(weather_df, left_index=True, right_index=True, how='inner')
    return df


def load_and_prepare_multivariate_data(
    load_path,
    weather_path,
    freq='H'
):
    """
    Pipeline hoàn chỉnh:
    - Load electricity
    - Load weather
    - Resample
    - Merge
    """

    load_df = load_electricity_data(load_path)
    weather_df = load_weather_data(weather_path)

    load_df = resample_data(load_df, freq)
    weather_df = resample_data(weather_df, freq)

    df = merge_load_weather(load_df, weather_df)

    return df

import matplotlib.pyplot as plt

