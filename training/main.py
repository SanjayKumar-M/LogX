import pandas as pd
df = pd.read_csv('dataset/synthetic_logs.csv')
print(df.target_label.unique())