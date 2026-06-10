import pandas as pd
df=pd.read_csv('../data/results.csv')
df=df.dropna(subset=['home_score','away_score'])
df['date']=pd.to_datetime(df['date'])
# Remove friendlies coz they aint reflect real competition
df=df[df['tournament']!='Friendly']
# Keep only matches from 1990 onwards coz modern football era
df = df[df['date'].dt.year >= 1990]
# Add result column from home team perspective
# 1 = home win, 0 = draw, 2 = away win
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] == row['away_score']:
        return 0
    else:
        return 2
df['result'] = df.apply(get_result, axis=1)
# Save cleaned data
df.to_csv('../data/cleaned_results.csv', index=False)

print("Original shape:", pd.read_csv('../data/results.csv').shape)
print("Cleaned shape:", df.shape)
print("\nResult distribution:")
print(df['result'].value_counts())
print("\nDate range after cleaning:")
print("From:", df['date'].min())
print("To:", df['date'].max())

