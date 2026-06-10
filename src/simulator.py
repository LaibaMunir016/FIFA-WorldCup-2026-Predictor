import pandas as pd
import numpy as np
import pickle
import random

# Load model
with open('src/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load features for team stats
df = pd.read_csv('data/features.csv')

# Get latest stats for each team
def get_team_features(team):
    home = df[df['home_team'] == team].tail(10)
    away = df[df['away_team'] == team].tail(10)

    if len(home) == 0 and len(away) == 0:
        return {
            'win_rate': 0.4, 'draw_rate': 0.25,
            'goals_scored': 1.2, 'goals_conceded': 1.2, 'form': 0.4
        }

    win_rate = home['home_win_rate'].mean() if len(home) > 0 else away['away_win_rate'].mean()
    draw_rate = home['home_draw_rate'].mean() if len(home) > 0 else away['away_draw_rate'].mean()
    goals_scored = home['home_goals_scored'].mean() if len(home) > 0 else away['away_goals_scored'].mean()
    goals_conceded = home['home_goals_conceded'].mean() if len(home) > 0 else away['away_goals_conceded'].mean()
    form = home['home_form'].mean() if len(home) > 0 else away['away_form'].mean()

    return {
        'win_rate': win_rate, 'draw_rate': draw_rate,
        'goals_scored': goals_scored, 'goals_conceded': goals_conceded,
        'form': form
    }

# Predict match winner
def predict_match(home_team, away_team):
    h = get_team_features(home_team)
    a = get_team_features(away_team)

    features = [[
        h['win_rate'], a['win_rate'],
        h['draw_rate'], a['draw_rate'],
        h['goals_scored'], a['goals_scored'],
        h['goals_conceded'], a['goals_conceded'],
        h['form'], a['form']
    ]]

    proba = model.predict_proba(features)[0]
    return proba  # [draw, home_win, away_win]

# Simulate knockout match (no draws allowed)
def knockout_winner(team1, team2):
    proba = predict_match(team1, team2)
    draw_prob, home_win_prob, away_win_prob = proba

    # In knockout redistribute draw probability
    total = home_win_prob + away_win_prob
    home_win_prob = home_win_prob / total
    away_win_prob = away_win_prob / total

    return random.choices([team1, team2], weights=[home_win_prob, away_win_prob])[0]

# 2026 World Cup Groups (48 teams, 12 groups)
groups = {
    'A': ['USA', 'Mexico', 'Canada', 'Uruguay'],
    'B': ['Argentina', 'Chile', 'Peru', 'Australia'],
    'C': ['Brazil', 'Colombia', 'Ecuador', 'Japan'],
    'D': ['France', 'Belgium', 'Morocco', 'Croatia'],
    'E': ['England', 'Netherlands', 'Senegal', 'Iran'],
    'F': ['Spain', 'Portugal', 'South Korea', 'Costa Rica'],
    'G': ['Germany', 'Switzerland', 'Serbia', 'Cameroon'],
    'H': ['Italy', 'Poland', 'Tunisia', 'Qatar'],
    'I': ['Denmark', 'Sweden', 'Nigeria', 'Ghana'],
    'J': ['Turkey', 'Ukraine', 'Algeria', 'Saudi Arabia'],
    'K': ['Egypt', 'Ivory Coast', 'Czechia', 'Bolivia'],
    'L': ['New Zealand', 'Panama', 'Venezuela', 'Paraguay']
}

# Simulate group stage - top 2 from each group qualify
def simulate_group_stage():
    qualifiers = []
    for group, teams in groups.items():
        points = {team: 0 for team in teams}

        # Each team plays others once
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                proba = predict_match(teams[i], teams[j])
                draw_p, home_p, away_p = proba
                result = random.choices([0, 1, 2], weights=[draw_p, home_p, away_p])[0]

                if result == 1:
                    points[teams[i]] += 3
                elif result == 2:
                    points[teams[j]] += 3
                else:
                    points[teams[i]] += 1
                    points[teams[j]] += 1

        # Top 2 qualify
        sorted_teams = sorted(points, key=points.get, reverse=True)
        qualifiers.extend(sorted_teams[:2])

    return qualifiers  # 24 teams

# Simulate full tournament
def simulate_tournament():
    qualified = simulate_group_stage()  # 24 teams
    random.shuffle(qualified)

    # Round of 32 → not needed, go straight knockouts
    round_of_16 = qualified  # 24 teams play knockout

    # Keep playing until 1 winner
    current_round = round_of_16
    while len(current_round) > 1:
        next_round = []
        random.shuffle(current_round)
        for i in range(0, len(current_round)-1, 2):
            winner = knockout_winner(current_round[i], current_round[i+1])
            next_round.append(winner)
        if len(current_round) % 2 != 0:
            next_round.append(current_round[-1])
        current_round = next_round

    return current_round[0]

# Run 1000 simulations
print("Running 1000 tournament simulations...")
win_counts = {}
simulations = 1000

for i in range(simulations):
    if i % 100 == 0:
        print(f"Simulation {i}/1000...")
    winner = simulate_tournament()
    win_counts[winner] = win_counts.get(winner, 0) + 1

# Sort by win count
sorted_winners = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

print("\n🏆 FIFA World Cup 2026 Predictions (based on 1000 simulations)")
print("-" * 50)
for team, wins in sorted_winners[:10]:
    probability = round((wins / simulations) * 100, 1)
    bar = "█" * int(probability / 2)
    print(f"{team:<20} {probability:>5}%  {bar}")