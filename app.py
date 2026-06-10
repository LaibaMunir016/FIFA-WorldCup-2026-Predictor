import streamlit as st
import pandas as pd
import pickle
import random
import sys
sys.path.append('/src')
from src.simulator import (
    predict_match,
    knockout_winner,
    simulate_tournament,
    get_team_features,
    groups
)
# Page config
st.set_page_config(
    page_title="FIFA 2026 Predictor",
    page_icon="🏆",
    layout="centered"
)

st.title("🏆 FIFA World Cup 2026 Predictor")
st.caption("ML-powered predictions using 49,000+ historical matches")

# --- SECTION 1: Tournament Winner Prediction ---
st.header("🌍 Tournament Winner Simulation")
st.write("Simulate the entire 2026 World Cup tournament using our ML model.")

num_sims = st.slider("Number of simulations", min_value=100, max_value=1000, value=500, step=100)

if st.button("Run Tournament Simulation ↗"):
    with st.spinner("Simulating tournament..."):
        win_counts = {}
        for i in range(num_sims):
            winner = simulate_tournament()
            win_counts[winner] = win_counts.get(winner, 0) + 1

        sorted_winners = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
        top10 = sorted_winners[:10]

        teams = [t[0] for t in top10]
        probs = [round((t[1] / num_sims) * 100, 1) for t in top10]

        results_df = pd.DataFrame({'Team': teams, 'Win Probability (%)': probs})
        results_df.index += 1

        st.subheader("Top 10 Predicted Winners")
        st.bar_chart(results_df.set_index('Team'))
        st.dataframe(results_df, use_container_width=True)
        st.success(f"🏆 Most likely winner: {teams[0]} ({probs[0]}%)")

# --- SECTION 2: Head to Head Prediction ---
st.header("⚽ Head-to-Head Match Predictor")
st.write("Select any two teams to predict the match outcome.")

all_teams = sorted(list(set(
    [team for group in groups.values() for team in group]
)))

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", all_teams, index=all_teams.index("Brazil"))
with col2:
    away_team = st.selectbox("Away Team", all_teams, index=all_teams.index("France"))

if st.button("Predict Match ↗"):
    if home_team == away_team:
        st.warning("Please select two different teams!")
    else:
        proba = predict_match(home_team, away_team)
        draw_p, home_p, away_p = proba

        st.subheader(f"{home_team} vs {away_team}")

        col1, col2, col3 = st.columns(3)
        col1.metric(f"{home_team} Win", f"{round(home_p*100, 1)}%")
        col2.metric("Draw", f"{round(draw_p*100, 1)}%")
        col3.metric(f"{away_team} Win", f"{round(away_p*100, 1)}%")

        if home_p > away_p and home_p > draw_p:
            st.success(f"✅ Prediction: {home_team} wins")
        elif away_p > home_p and away_p > draw_p:
            st.success(f"✅ Prediction: {away_team} wins")
        else:
            st.info("🤝 Prediction: Draw")

# --- SECTION 3: Team Stats ---
st.header("📊 Team Statistics")
selected_team = st.selectbox("Select a team", all_teams)

stats = get_team_features(selected_team)
col1, col2, col3 = st.columns(3)
col1.metric("Win Rate", f"{round(stats['win_rate']*100, 1)}%")
col2.metric("Avg Goals Scored", stats['goals_scored'])
col3.metric("Recent Form", f"{round(stats['form']*100, 1)}%")

# Footer
st.divider()
st.caption("Built with Python, scikit-learn, XGBoost & Streamlit | Dataset: Mart Jürisoo (Kaggle)")