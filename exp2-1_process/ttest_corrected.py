import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

import glob


# 讀取資料
df = pd.read_csv('C:\\codehome\\eegdata\\output\\6\\6_power_change.csv')
results = []

results = []

grouped = df.groupby(['band', 'region'])

for (band, region), group in grouped:
    if len(group) > 1:
        stat, pvalue = ttest_rel(group['pre_power'], group['post_power'])
    else:
        stat, pvalue = float('nan'), float('nan')
    results.append({'band': band, 'region': region, 't_statistic': stat, 'p_value': pvalue})

results_df = pd.DataFrame(results)

valid_pvals = results_df['p_value'].dropna()
reject, pvals_corrected, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')

results_df.loc[results_df['p_value'].notna(), 'p_value_corrected'] = pvals_corrected
results_df.loc[results_df['p_value'].notna(), 'reject_null'] = reject

# results_df.to_csv('C:\\codehome\\eegdata\\output\\all_ttest_corrected.csv', index=False)

print(results_df)

plt.figure(figsize=(12, 6))

bands = results_df['band'].unique()
height = 0.8  
spacing_between_bands = 0  
spacing_within_band = 1  

y_pos = []
band_start = 0  

for band in bands:
    band_rows = results_df[results_df['band'] == band]
    band_height = len(band_rows)
    
    y_pos.extend(range(band_start, band_start + band_height))
    
    band_start += band_height + spacing_within_band

    band_start += spacing_between_bands

plt.barh(y_pos, results_df['p_value_corrected'].fillna(1), height=height)
# plt.barh(y_pos, results_df['p_value_corrected'].fillna(1), height=height, color='#6eb3c5')
plt.yticks(y_pos, results_df['band'] + '-' + results_df['region'])

plt.xlabel('Corrected p-value')
plt.title('Score')

plt.axvline(0.05, color='red', linestyle='--', label='Significance threshold (0.05)')
plt.legend()

plt.tight_layout()

plt.show()