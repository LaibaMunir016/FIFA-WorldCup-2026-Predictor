# 🏆 FIFA World Cup 2026 Predictor

An end-to-end Machine Learning project that predicts the FIFA World Cup 2026 
winner using 49,000+ historical international football matches. The model uses 
XGBoost classifier combined with Monte Carlo simulation to forecast tournament 
outcomes.

---

## 📊 Dataset
- **Source:** Mart Jürisoo — Kaggle
- **Link:** https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- **Original records:** 49,450 international matches (1872–2026)
- **After cleaning:** 21,451 competitive matches (1990–2024)
- **Files used:** results.csv, shootouts.csv

---

## ⚙️ Project Pipeline

### 1. Data Cleaning
- Removed 72 rows with missing scores
- Removed 18,000+ friendly matches (not competitive)
- Kept only matches from 1990 onwards (modern football era)
- Removed defunct nations like Soviet Union, Yugoslavia

### 2. Feature Engineering
For every match, 5 features were calculated per team using only 
historical data before that match date (no data leakage):

| Feature | Description |
|---------|-------------|
| Win Rate | % of matches won historically |
| Draw Rate | % of matches drawn historically |
| Avg Goals Scored | Average goals scored per match |
| Avg Goals Conceded | Average goals conceded per match |
| Recent Form | Win rate in last 5 matches |

### 3. Model Training
Two models were trained and compared:

| Model | Accuracy |
|-------|---------|
| Logistic Regression (baseline) | 59.3% |
| XGBoost (final model) | 59.7% |

**Note:** 60% accuracy is strong for football prediction — even 
professional analysts rarely exceed 65% due to football's 
unpredictable nature.

### 4. Class Imbalance Observation
| Result | Count | % |
|--------|-------|---|
| Home Win | 10,506 | 49% |
| Away Win | 6,300 | 29% |
| Draw | 4,645 | 22% |

Draws were hardest to predict due to class imbalance — 
a known limitation acknowledged in this project.

### 5. Tournament Simulation (Monte Carlo)
- Simulated full 2026 World Cup bracket (48 teams, 12 groups)
- Group stage + knockout rounds simulated
- Run 1000 times to get win probabilities per team

---

## 🏆 Top Predicted Winners
| Rank | Team | Win Probability |
|------|------|----------------|
| 1 | Spain | 19.3% |
| 2 | Portugal | 7.4% |
| 3 | England | 6.9% |
| 4 | France | 6.9% |
| 5 | Australia | 5.9% |
| 6 | Italy | 5.5% |
| 7 | Iran | 5.3% |
| 8 | Nigeria | 5.1% |
| 9 | Brazil | 4.8% |
| 10 | Morocco | 4.3% |

---

## 🖥️ Streamlit App Features

### 🌍 Tournament Winner Simulation
- Adjust number of simulations (100–1000) using a slider
- Runs full Monte Carlo tournament simulation
- Displays top 10 predicted winners as bar chart and table

### ⚽ Head-to-Head Match Predictor
- Select any 2 teams from 2026 World Cup
- Predicts win/draw/loss probability for each team
- Shows clear winner prediction

### 📊 Team Statistics
- Select any team to view their historical stats
- Shows win rate, average goals scored, recent form

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas & numpy | Data manipulation |
| scikit-learn | Logistic Regression, train/test split, evaluation |
| XGBoost | Main prediction model |
| Streamlit | Web app interface |
| Git & GitHub | Version control |

---

## 🚀 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/LaibaMunir016/FIFA-WorldCup-2026-Predictor.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚠️ Known Limitations & Future Improvements
| Limitation | Future Fix |
|-----------|-----------|
| Draws hard to predict (class imbalance) | Use SMOTE oversampling |
| No ELO ratings used | Add dynamic ELO rating system |
| Recent form only last 5 matches | Extend to last 10 matches |
| No player-level data | Add FIFA player ratings |
| No injury/suspension data | Integrate real-time team news |

---

## 👩‍💻 Author
**Laiba Munir**  
GitHub: [@LaibaMunir016](https://github.com/LaibaMunir016)
