import numpy as np
import pandas as pd
import mne
import pickle
from pylsl import resolve_streams, StreamInlet
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# ---------- 載入模型與參數 ----------
model_path = '1-1svm_model.pkl'
scaler_path = '1-1scaler.pkl'
feature_columns_path = '1-1feature_columns.pkl'
output_csv = '1-6emotion_detection_log.csv'

img_positive = mpimg.imread('1.png')
img_negative = mpimg.imread('2.png')

# 讀取ICA物件 (記得改成你的檔案路徑)
ica_path = '1_ica.fif'
ica = mne.preprocessing.read_ica(ica_path)
ica.exclude = [1, 3, 5]  # 依你設定

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
feature_columns = pickle.load(open(feature_columns_path, 'rb'))

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

# print(len(ica.ch_names))  # ICA訓練時的通道數
# print(ica.ch_names)


# 頻道名稱（同之前）
ch_names = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
    'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'A1', 'T5',
    'P3', 'Pz', 'P4', 'T6', 'A2', 'O1', 'Oz', 'O2', 'HEOL', 'HEOR', 'VEOL', 'VEOR',
    'MGFP'
]

exclude_channels = ['HEOL', 'HEOR', 'VEOL', 'VEOR', 'MGFP']
keep_indices = [i for i, ch in enumerate(ch_names) if ch not in exclude_channels]
keep_ch_names = [ch_names[i] for i in keep_indices]

sfreq = None  # 會從LSL設定

window_sec = 4
step_sec = 1


if not os.path.isfile(output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_sec', 'Detection'])  # 表頭

def get_freqs_idx(psd_freqs, fmin, fmax):
    idx_min = (np.abs(psd_freqs - fmin)).argmin()
    idx_max = (np.abs(psd_freqs - fmax)).argmin()
    if idx_min > idx_max:
        idx_min, idx_max = idx_max, idx_min
    return np.arange(idx_min, idx_max + 1)

def calc_band_region_power_from_psd(psd, psd_freqs):
    features = []
    for band, (fmin, fmax) in freq_bands.items():
        freq_idx = get_freqs_idx(psd_freqs, fmin, fmax)
        for region, chans in channel_groups.items():
            band_power = psd[np.ix_(chans, freq_idx)].mean()
            features.append((band, region, band_power))
    return features

def main():
    print("尋找 LSL EEG 資料流...")
    streams = resolve_streams(wait_time=2)
    eeg_streams = [s for s in streams if s.type() == 'EEG']

    if not eeg_streams:
        raise RuntimeError("找不到 EEG 流")

    inlet = StreamInlet(eeg_streams[0])
    global sfreq
    sfreq = int(eeg_streams[0].nominal_srate())
    n_channels = eeg_streams[0].channel_count()

    print(f"連線成功: 頻率 {sfreq} Hz, 頻道數 {n_channels}")

    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    data_buffer = np.zeros((len(keep_indices), window_samples))
    total_samples = 0
    last_detection = None

    # 建立info物件用於RawArray
    info = mne.create_info(ch_names=keep_ch_names, sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage)

    plt.ion()
    fig, ax = plt.subplots()
    img_handle = ax.imshow(img_positive)
    ax.axis('off')

    while True:
        samples = []
        for _ in range(step_samples):
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                samples.append(sample)

        if len(samples) < step_samples:
            print("資料不足，等待中...")
            continue

        samples_np = np.array(samples)  # shape (step_samples, 原始通道數)
        samples_np = samples_np[:, keep_indices].T  # shape (channels, samples)

        data_buffer = np.roll(data_buffer, -step_samples, axis=1)
        data_buffer[:, -step_samples:] = samples_np
        total_samples += step_samples

        if np.count_nonzero(data_buffer) == data_buffer.size:
            # 轉成RawArray，套用ICA
            raw_segment = mne.io.RawArray(data_buffer, info)
            raw_clean = ica.apply(raw_segment.copy())

            # 用Welch方法計算PSD
            psd_obj = raw_clean.compute_psd(fmin=1, fmax=50, n_fft=int(sfreq*2), n_overlap=0, method='welch')
            psd = psd_obj.get_data()  # channels x freqs
            psd_freqs = psd_obj.freqs

            # 分成前後半段功率，因為之前是做差
            half_samples = window_samples // 2
            pre_data = data_buffer[:, :half_samples]
            post_data = data_buffer[:, half_samples:]

            raw_pre = mne.io.RawArray(pre_data, info)
            raw_pre = ica.apply(raw_pre.copy())
            psd_pre_obj = raw_pre.compute_psd(fmin=1, fmax=50, n_fft=int(sfreq*2), n_overlap=0, method='welch')
            psd_pre = psd_pre_obj.get_data()

            raw_post = mne.io.RawArray(post_data, info)
            raw_post = ica.apply(raw_post.copy())
            psd_post_obj = raw_post.compute_psd(fmin=1, fmax=50, n_fft=int(sfreq*2), n_overlap=0, method='welch')
            psd_post = psd_post_obj.get_data()

            pre_features = calc_band_region_power_from_psd(psd_pre, psd_freqs)
            post_features = calc_band_region_power_from_psd(psd_post, psd_freqs)

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
            df_band = pd.get_dummies(df_feat['band'], prefix='band')
            df_region = pd.get_dummies(df_feat['region'], prefix='region')
            X_num = df_feat[['pre_power', 'post_power', 'power_change']]
            X = pd.concat([X_num, df_band, df_region], axis=1)

            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_columns]

            X_values = scaler.transform(X.values)
            y_pred = model.predict(X_values)

            start_time_sec = total_samples / sfreq - window_sec

            if np.any(y_pred == 1):
                detection = "負面情緒"
                print(f"{start_time_sec:.2f}秒：警告！偵測到負面情緒！")
                if last_detection != detection:
                    img_handle.set_data(img_negative)
            else:
                detection = "情緒正常"
                print(f"{start_time_sec:.2f}秒：情緒正常。")
                if last_detection != detection:
                    img_handle.set_data(img_positive)

            # 只在狀態改變時寫CSV
            if detection != last_detection:
                with open(output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([start_time_sec, detection])
                last_detection = detection

            fig.canvas.draw()
            fig.canvas.flush_events()

if __name__ == '__main__':
    main()
