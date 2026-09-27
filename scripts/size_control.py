# Size control for the matched evaluations: random, class-balanced subsets of the main sets with the same
# size as the age x sex matched subsets (10 draws). Run from notebooks/: python ../scripts/size_control.py
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
nb=json.load(open('model_training_v8.ipynb'))
codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g={'display':lambda *a,**k: None}
for s in codes:
    if 'BINARY_SETTINGS' in s: break
    exec(s,g)
df, patho_main, bin_m6, multi_m = g['df'], g['patho_main'], g['bin_matched_6'], g['multi_matched']
num, cat = g['FEATURE_SETS']['features only']; tp, lp = g['build_preprocessors'](num, cat)
def strat_sub(data, label, n_per, seed):
    return pd.concat([x.sample(n=n_per, random_state=seed) for _, x in data.groupby(label)])
rows=[]
SEEDS=range(10)
for seed in SEEDS:
    sb = strat_sub(df, 'is_healthy', len(bin_m6)//2, seed)
    for mn, m in g['binary_models'].items():
        f = pd.DataFrame(g['cv_binary'](m, sb[num+cat], sb.is_healthy.astype(int), sb.speaker_id.astype(str), tp, lp)[0])
        rows.append(('binary', 'random size-matched', seed, mn, f.f1_macro.mean()))
    sm = strat_sub(patho_main, 'target_label', len(multi_m)//2, seed)
    for mn, m in g['multi_models'].items():
        f = pd.DataFrame(g['cv_multiclass'](m, sm[num+cat], sm.target_label.astype(str), sm.speaker_id.astype(str), tp, lp)[0])
        rows.append(('disease group', 'random size-matched', seed, mn, f.f1_macro.mean()))
    print('seed', seed, 'done', flush=True)
R=pd.DataFrame(rows, columns=['task','setting','seed','model','f1'])
R.to_csv('../results/main_v8/size_control.csv', index=False)
per_model = R.groupby(['task','model']).f1.agg(['mean','std']).round(3)
print(per_model)
best = per_model.reset_index().sort_values('mean', ascending=False).drop_duplicates('task')
print(best)
print('n binary sub', len(bin_m6), 'n disease sub', len(multi_m))
print('age of patho in matched binary:', bin_m6[~bin_m6.is_healthy.astype(bool)].age.mean().round(1), 'vs main patho', df[~df.is_healthy.astype(bool)].age.mean().round(1))
print('disease mix matched binary:'); print(bin_m6[~bin_m6.is_healthy.astype(bool)].target_label.value_counts(normalize=True).round(2))
print('disease mix main:'); print(df[~df.is_healthy.astype(bool)].target_label.value_counts(normalize=True).round(2))
