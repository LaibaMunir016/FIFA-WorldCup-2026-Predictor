import pandas as pd
import numpy as np

df = pd.read_csv('../data/cleaned_results.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# --- FEATURE 1: Win rate for each team ---
def get_team_stats(df, team):
    home = df[df['home_team'] == team]
    away = df[df['away_team'] == team]

    home_wins = (home['result'] == 1).sum()
    away_wins = (away['result'] == 2).sum()
    home_draws = (home['result'] == 0).sum()
    away_draws = (away['result'] == 0).sum()
    total = len(home) + len(away)

    if total == 0:
        return 0, 0

    win_rate = (home_wins + away_wins) / total
    draw_rate = (home_draws + away_draws) / total
    return win_rate, draw_rate

# --- FEATURE 2: Average goals scored and conceded ---
def get_goal_stats(df, team):
    home = df[df['home_team'] == team]
    away = df[df['away_team'] == team]

    goals_scored = pd.concat([home['home_score'], away['away_score']]).mean()
    goals_conceded = pd.concat([home['away_score'], away['home_score']]).mean()

    return round(goals_scored, 2), round(goals_conceded, 2)

# --- FEATURE 3: Recent form (last 5 matches) ---
def get_recent_form(df, team, before_date):
    mask = ((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < before_date)
    recent = df[mask].tail(5)

    if len(recent) == 0:
        return 0.5

    wins = 0
    for _, row in recent.iterrows():
        if row['home_team'] == team and row['result'] == 1:
            wins += 1
        elif row['away_team'] == team and row['result'] == 2:
            wins += 1

    return round(wins / len(recent), 2)

# --- BUILD FEATURE MATRIX ---
print("Building features for each match...")

rows = []
for _, match in df.iterrows():
    home = match['home_team']
    away = match['away_team']
    date = match['date']

    home_wr, home_dr = get_team_stats(df[df['date'] < date], home)
    away_wr, away_dr = get_team_stats(df[df['date'] < date], away)

    home_gs, home_gc = get_goal_stats(df[df['date'] < date], home)
    away_gs, away_gc = get_goal_stats(df[df['date'] < date], away)

    home_form = get_recent_form(df, home, date)
    away_form = get_recent_form(df, away, date)

    rows.append({
        'date': date,
        'home_team': home,
        'away_team': away,
        'home_win_rate': home_wr,
        'away_win_rate': away_wr,
        'home_draw_rate': home_dr,
        'away_draw_rate': away_dr,
        'home_goals_scored': home_gs,
        'away_goals_scored': away_gs,
        'home_goals_conceded': home_gc,
        'away_goals_conceded': away_gc,
        'home_form': home_form,
        'away_form': away_form,
        'result': match['result']
    })

features_df = pd.DataFrame(rows)
features_df.to_csv('../data/features.csv', index=False)
print("Done! Shape:", features_df.shape)
print(features_df.head())