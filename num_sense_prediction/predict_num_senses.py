import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from generate_profiles import generate

df_train = pd.read_csv("profiled_words.tsv", sep='\t')  # Averaged num of senses
df = pd.read_csv("sem_cluster_data.tsv", sep='\t')
df_test = df.groupby('word').gold_sense_id.nunique().reset_index()
df_test.rename(columns={'word': 'Lemma'}, inplace=True)
df_test.to_csv("sem_cluster_unique.tsv", sep='\t', index=False)
generate("sem_cluster_unique.tsv", "sem_cluster_unique")
df_train = df_train.loc[~df_train.Lemma.isin(set(df_test.Lemma.tolist()))]
X, y = df_train.loc[:, 'Anim': 'nummod_child'].to_numpy(), df_train.Mean.to_numpy()
parameters = {'max_depth': range(3, 20)}
reg = GridSearchCV(DecisionTreeRegressor(), parameters, n_jobs=4)
reg.fit(X=X, y=y)
tree_model = reg.best_estimator_
df_test = pd.read_csv("profiled_sem_cluster_unique.tsv", sep='\t')
df_test.drop(columns=['gold_sense_id'], inplace=True)
df_test['num_senses'] = tree_model.predict(df_test.loc[:, 'Anim': 'nummod_child'].to_numpy())
df_test['num_senses'] = df_test['num_senses'].astype('int64')
df = df.merge(df_test, left_on='word', right_on='Lemma')
df.drop(columns=['Lemma'], inplace=True)
df.to_csv('num_senses_sem_cluster_data.tsv', sep='\t', index=False)
