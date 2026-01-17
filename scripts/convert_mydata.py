import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, asin, sqrt

# Configuration
IN_CSV = './data/my_data/dataset.csv'
OUT_DIR = './data/my_data'
OUT_NAME = 'my_data'
# Put the target feature `Flow_Volume` first so LEAF treats it as the output (`out_dim=1`).
FEATURE_COLS = ['Flow_Volume', 'Avg_Occupancy_%', 'Traffic_Density']
TIME_COL = 'Timestamp'
DATE_COL = 'Date'
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
FREQ = '10S'  # 10 seconds

os.makedirs(OUT_DIR, exist_ok=True)

# Haversine distance in kilometers
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(min(1, sqrt(a)))
    km = 6371 * c
    return km

print('Reading', IN_CSV)
df = pd.read_csv(IN_CSV)
# combine date and time into datetime; use dayfirst because Date column appears dd/mm/YYYY

df['datetime'] = pd.to_datetime(df[DATE_COL].astype(str) + ' ' + df[TIME_COL].astype(str), dayfirst=True, errors='coerce')
if df['datetime'].isnull().any():
    raise RuntimeError('Some datetime values could not be parsed; inspect your CSV DATE/Timestamp format')

# Identify unique sensors by (lat,lon)
df['lat_round'] = df[LAT_COL].round(6)
df['lon_round'] = df[LON_COL].round(6)
coords = df[['lat_round', 'lon_round']].drop_duplicates().reset_index(drop=True)
coords['node_idx'] = range(len(coords))
coord_to_idx = { (row.lat_round, row.lon_round): int(row.node_idx) for row in coords.itertuples() }

print('Found', len(coords), 'unique sensors (nodes)')

df['node_idx'] = df.apply(lambda r: coord_to_idx[(round(r[LAT_COL],6), round(r[LON_COL],6))], axis=1)

# prepare global time index (5-min frequency)
start = df['datetime'].min().floor(FREQ)
end = df['datetime'].max().ceil(FREQ)
time_index = pd.date_range(start, end, freq=FREQ)
print('Time range:', start, '->', end, '(', len(time_index), 'steps)')

num_nodes = len(coords)
num_steps = len(time_index)
in_dim = len(FEATURE_COLS)

# allocate array (num_steps, num_nodes, in_dim)
arr = np.zeros((num_steps, num_nodes, in_dim), dtype=np.float32)

# for each node, resample to 5T using mean and fill missing with 0
for node in range(num_nodes):
    node_df = df[df['node_idx']==node].set_index('datetime').sort_index()
    if node_df.shape[0] == 0:
        continue
    s = node_df[FEATURE_COLS].resample(FREQ).mean()
    s = s.reindex(time_index).fillna(0)
    if s.shape[0] != num_steps:
        s = s.reindex(time_index).fillna(0)
    arr[:, node, :] = s.values.astype(np.float32)

# save npz
out_npz = os.path.join(OUT_DIR, f'{OUT_NAME}.npz')
np.savez(out_npz, data=arr)
print('Saved time-series to', out_npz, 'with shape', arr.shape)

# build distance CSV using haversine between sensor coords
out_csv = os.path.join(OUT_DIR, f'{OUT_NAME}.csv')
with open(out_csv, 'w') as f:
    for i in range(num_nodes):
        lat1 = float(coords.loc[i, 'lat_round'])
        lon1 = float(coords.loc[i, 'lon_round'])
        for j in range(i+1, num_nodes):
            lat2 = float(coords.loc[j, 'lat_round'])
            lon2 = float(coords.loc[j, 'lon_round'])
            d = haversine(lat1, lon1, lat2, lon2)
            f.write(f"{i},{j},{d}\n")
print('Saved distance CSV to', out_csv)

# Also save node order mapping (optional)
node_order_path = os.path.join(OUT_DIR, f'{OUT_NAME}.txt')
with open(node_order_path, 'w') as f:
    for i in range(num_nodes):
        lat = coords.loc[i, 'lat_round']
        lon = coords.loc[i, 'lon_round']
        f.write(f"{lat},{lon}\n")
print('Saved node order (lat,lon) to', node_order_path)

print('Done.')
