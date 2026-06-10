import pandas as pd

# Load the dataset
df = pd.read_csv('../data/results.csv')

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nNull values:")
print(df.isnull().sum())
print("\nDate range:")
print("Earliest:", df['date'].min())
print("Latest:", df['date'].max())
print("\nTotal unique teams:", df['home_team'].nunique())
print("\nTournaments:")
print(df['tournament'].value_counts().head(10))