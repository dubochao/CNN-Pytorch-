import pandas as pd
df = pd.read_csv('data/weibo_senti_100k.csv', encoding='utf-8')
# df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本
df = df.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.1 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
#dataframe = pd.DataFrame(df_train)
df_test.to_csv("data/test.csv",sep=',')
df_train.to_csv("data/train.csv",sep=',')
print (   df_test.shape, df_train.shape) # (3184, 12) (318, 12) (2866, 12)

