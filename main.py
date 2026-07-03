import pandas as pd
import numpy as np
import os, glob, warnings, time
warnings.filterwarnings('ignore')

t0 = time.time()

def find_file(name):
    if os.path.exists(name): return name
    m = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if m: return m[0]
    raise FileNotFoundError(name)

TRAIN = find_file('train.csv')
SUB   = find_file('sample_submission.csv')


# data cleaning
raw = pd.read_csv(TRAIN,
    usecols=['Date', 'ItemCode', 'Quantity', 'SalesAmount'],
    low_memory=False)
raw['Date']     = pd.to_datetime(raw['Date'])
raw['Quantity'] = pd.to_numeric(raw['Quantity'], errors='coerce').fillna(0)
raw['SalesAmount'] = pd.to_numeric(
    raw['SalesAmount'].astype(str).str.replace(r'[^\d.\-]', '', regex=True),
    errors='coerce').fillna(0)

rev_by_sku = raw.groupby('ItemCode')['SalesAmount'].sum().clip(lower=0)

daily = (raw.groupby(['Date', 'ItemCode'])['Quantity']
            .sum().clip(lower=0).reset_index())


# pivot matrix transformation
pivot = daily.pivot_table(
    index='Date', columns='ItemCode', values='Quantity', fill_value=0)
all_dates = pd.date_range(pivot.index.min(), pivot.index.max(), freq='D')
pivot     = pivot.reindex(all_dates, fill_value=0)

skus           = np.array(pivot.columns.tolist())
vals           = pivot.values.astype(np.float32)   
n_days, n_skus = vals.shape
rev_arr        = np.array([rev_by_sku.get(s, 0) for s in skus], dtype=np.float32)


# SKU classification
freq90   = (vals[-90:] > 0).mean(axis=0)
dense_m  = freq90 >= 0.40      
medium_m = (freq90 >= 0.10) & (~dense_m)  

dead_m   = (vals[-45:] > 0).sum(axis=0) == 0

TOP_N    = 75
top_idx  = np.argsort(rev_arr)[::-1][:TOP_N]
top_m    = np.zeros(n_skus, dtype=bool)
top_m[top_idx] = True


# weekly cycle features
wd_vals   = pd.DatetimeIndex(all_dates).dayofweek.values
wday_mean = np.zeros((7, n_skus), dtype=np.float32)
for wd in range(7):
    mask = wd_vals[-365:] == wd
    cnt  = mask.sum()
    wday_mean[wd] = vals[-365:][mask].sum(axis=0) / (cnt + 1e-8)

wf_raw = wday_mean / (vals[-365:].mean(axis=0) + 1e-8)
wday_factor = np.clip(0.9 + 0.1 * wf_raw, 0.0, 3.0).astype(np.float32)


# calculate baseline for each group
mean28 = vals[-28:].mean(axis=0).astype(np.float32)
mean84 = vals[-84:].mean(axis=0).astype(np.float32)

base_dense = vals[-21:].mean(axis=0).astype(np.float32)   

cv = vals[-28:].std(axis=0) / (mean28 + 1e-8)             
base_medium = np.where(
    (cv > 1.5) & medium_m,                    
    0.7 * mean28 + 0.3 * mean84,             
    0.3 * mean28 + 0.7 * mean84              
).astype(np.float32)

base_top = vals[-28:].mean(axis=0).astype(np.float32)

recent_max  = vals[-28:].max(axis=0).astype(np.float32)
cap_default = np.minimum(2.0 * recent_max, 3.0 * mean28)


# forecast horizon
HORIZON      = 56
future_dates = pd.date_range(all_dates[-1] + pd.Timedelta(days=1), periods=HORIZON)
future_wd    = future_dates.dayofweek.values

preds = np.zeros((HORIZON, n_skus), dtype=np.float32)

for i, wd in enumerate(future_wd):
    if wd == 6:          
        continue

    wf_day = wday_factor[wd]   

    preds[i, dense_m]  = base_dense[dense_m] * wf_day[dense_m]
    damped = 0.2 * wf_day[medium_m] + 0.8  
    preds[i, medium_m] = base_medium[medium_m] * damped
    td = top_m & dense_m
    tm = top_m & medium_m
    if td.any():
        preds[i, td] = base_top[td] * wf_day[td]
    if tm.any():
        preds[i, tm] = base_top[tm] * (0.2 * wf_day[tm] + 0.8)

preds = np.clip(preds, 0.0, None)
preds = np.minimum(preds, cap_default[None, :]) 
preds[:, dead_m] = 0.0                             


# packaging submission (validation & evaluation)
sku_idx    = {s: i for i, s in enumerate(skus)}
sub        = pd.read_csv(SUB)
fcols      = [f'F{i}' for i in range(1, 29)]
val_preds  = preds[:28, :]    
eval_preds = preds[28:, :]    

rows = []
for rid in sub['id']:
    sku, typ = rid.rsplit('_', 1)
    if sku in sku_idx:
        idx = sku_idx[sku]
        p28 = val_preds[:, idx] if typ == 'validation' else eval_preds[:, idx]
        rows.append([rid] + p28.tolist())
    else:
        rows.append([rid] + [0.0] * 28)

result = pd.DataFrame(rows, columns=['id'] + fcols)
result[fcols] = result[fcols].clip(lower=0)


# apply magic multiplier
result[fcols] = result[fcols] * 1.015


# export output file
out_path = 'submission.csv'
result.to_csv(out_path, index=False)


# notification completed
elapsed = time.time() - t0
nz = (result[fcols].sum(axis=1) > 0).sum()
print(f"\File created successfully: {out_path}")
print(f"   Execution time : {elapsed:.1f}s")
print(f"   Size: {result.shape}")
