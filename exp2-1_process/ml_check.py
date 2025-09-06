import sys
sys.path.append(r'C:\codehome\eegdata\curry-python-reader')
import curryreader as cr
import numpy as np
import mne
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -------- 參數設定 --------
cdt_path = r'C:\codehome\eegdata\Acq 2025_05_13_1602.cdt'
ica_path = r'C:\codehome\eegdata\3_ica.fif'
model_path = '3xgb_model.pkl'
scaler_path = '3scaler.pkl'
feature_columns_path = '3feature_columns.pkl'
img_positive = mpimg.imread('1.png')
img_negative = mpimg.imread('2.png')

# 頻段和腦區定義
freq_bands = {
    'delta': (1, 4),
    'theta': (5, 7),
    'alpha': (8, 12),
    'beta': (13, 28),
    'gamma': (30, 50),
}
channel_groups = {
    'frontal': [i-1 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
    'temporal_left': [i-1 for i in [7, 12, 17, 23]],
    'temporal_right': [i-1 for i in [11, 16, 21, 27]],
    'parietal': [i-1 for i in [18, 19, 20, 22, 24, 25, 26, 29]],
    'occipital': [i-1 for i in [30, 31, 32]],
    'all': list(range(32)),
}

window_sec = 4
overlap_sec = window_sec - 3

# -------- 載入模型、scaler、欄位 --------
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
feature_columns = pickle.load(open(feature_columns_path, 'rb'))

# -------- 讀檔並建立 RawArray --------
raw_data = cr.read(inputfilename=cdt_path, plotdata=0, verbosity=2)
data = np.array(raw_data['data'])
ch_names = raw_data['labels']
sfreq = raw_data['info']['samplingfreq']

data_32 = data[:, :32].T
ch_names_32 = ch_names[:32]

info_32 = mne.create_info(ch_names=ch_names_32, sfreq=sfreq, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1005')
info_32.set_montage(montage)
raw_32 = mne.io.RawArray(data_32, info_32)

# 濾波
raw_32.filter(1, 50, fir_design='firwin')

# 載入並套用 ICA
ica = mne.preprocessing.read_ica(ica_path)

ica.exclude = [0, 2]  # 你想移除第0、2、5號成分
raw_clean = ica.apply(raw_32.copy())

# -------- 功率計算與預測函式 --------
def calc_power(segment_data, sfreq):
    freqs = np.arange(1, 51)
    n_cycles = freqs / 2.
    power = mne.time_frequency.tfr_array_morlet(
        segment_data[np.newaxis, :, :], sfreq=sfreq,
        freqs=freqs, n_cycles=n_cycles, output='power', decim=1
    )[0]  # shape (channels, freqs, times)
    power_mean = power.mean(axis=2)  # 時間平均
    return power_mean, freqs

def calc_band_region_power(power_mean, freqs):
    features = []
    for band, (fmin, fmax) in freq_bands.items():
        freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        for region, chans in channel_groups.items():
            band_region_power = power_mean[np.ix_(chans, freq_idx)].mean()
            features.append((band, region, band_region_power))
    return features

def sliding_window_predict(raw, sfreq, window_sec, overlap_sec, model, scaler, feature_columns):
    window_samples = int(window_sec * sfreq)
    step_samples = int(1 * sfreq)  # 1秒步長固定

    data = raw.get_data()
    n_samples = data.shape[1]

    plt.ion()
    fig, ax = plt.subplots()
    img_handle = ax.imshow(img_positive)
    ax.axis('off')
    ####
    # start_sec = 734
    # start_sample = int(start_sec * sfreq)
    # for start in range(start_sample, n_samples - window_samples + 1, step_samples):

    for start in range(0, n_samples - window_samples + 1, step_samples):
        segment = data[:, start:start + window_samples]

        # 直接計算當前視窗的前半段和後半段
        pre_seg = segment[:, :int(window_samples / 2)]
        post_seg = segment[:, int(window_samples / 2):]

        # 計算 pre_power
        power_mean_pre, freqs = calc_power(pre_seg, sfreq)
        pre_features = calc_band_region_power(power_mean_pre, freqs)

        # 計算 post_power
        power_mean_post, freqs = calc_power(post_seg, sfreq)
        post_features = calc_band_region_power(power_mean_post, freqs)

        # 合併特徵，計算 power_change
        features = []
        for i in range(len(pre_features)):
            band, region, pre_power = pre_features[i]
            _, _, post_power = post_features[i]
            power_change = post_power - pre_power
            features.append({
                'band': band,
                'region': region,
                'pre_power': pre_power,
                'post_power': post_power,
                'power_change': power_change
            })

        df_feat = pd.DataFrame(features)

        # one-hot encoding
        df_band = pd.get_dummies(df_feat['band'], prefix='band')
        df_region = pd.get_dummies(df_feat['region'], prefix='region')
        X_num = df_feat[['pre_power', 'post_power', 'power_change']]
        X = pd.concat([X_num, df_band, df_region], axis=1)

        # 補齊缺少欄位
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]

        # print(f"窗口時間: {start / sfreq:.2f}秒")
        # print("特徵描述統計:")
        # print(df_feat[['pre_power', 'post_power', 'power_change']].describe())
        # print("預測特徵欄位：", list(X.columns))
        # print("X shape:", X.shape)

        # 用 scaler 標準化整個輸入矩陣
        X_values = scaler.transform(X.values)

        y_pred = model.predict(X_values)
        print("預測結果 (y_pred):", y_pred)

        if np.any(y_pred ==1):
            img_handle.set_data(img_negative)
            print(f"警告：於時間 {start / sfreq:.2f} 秒偵測到負面情緒！")
        else:
            img_handle.set_data(img_positive)
        print("-"*50)
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    sliding_window_predict(raw_clean, sfreq, window_sec, overlap_sec, model, scaler, feature_columns)
