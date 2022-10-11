import pandas as pd
import os

df_dict = {'features': [], 'max_ari': []}
for file in os.listdir('result'):
    if 'overall' in file:
        df_dict['features'].append(file[12:-4])
        df_hp = pd.read_csv(f'result/{file}', sep='\t')
        df_dict['max_ari'].append(df_hp.maxari.values[0])


df_overall = pd.DataFrame(df_dict).sort_values('max_ari', ascending=False, ignore_index=True)
df_overall.to_csv('result_comparison_bts_rnc_dist.tsv', sep='\t')
