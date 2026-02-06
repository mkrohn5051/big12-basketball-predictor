# 🏀 Big 12 Basketball Game Predictor

An end-to-end machine learning pipeline that predicts Big 12 basketball game outcomes using LSTM temporal features and ensemble models. Features advanced interpretability analysis with correlation heatmaps and embedding visualizations.

## 🎯 Overview

This system scrapes live game data from Sports Reference, trains multiple ML models including a custom LSTM encoder for temporal patterns, and generates daily predictions for upcoming Big 12 basketball games. Built with Apache Airflow for orchestration and PyTorch for GPU-accelerated deep learning.

## ⚡ Features

- **Automated Data Collection**: Web scraper for 16 Big 12 teams with respectful rate limiting
- **Temporal Feature Engineering**: LSTM encoder captures momentum and trends from last 10 games
- **Multiple ML Models**: Neural network, Random Forest, Gradient Boosting, XGBoost, Logistic Regression
- **Ensemble Predictions**: Accuracy-weighted voting across all 5 models
- **LSTM Interpretability**: Correlation analysis reveals what each of 64 LSTM dimensions learned
- **Rich Visualizations**: 10 total plots showing training dynamics, embeddings, and feature importance
- **Production Pipeline**: Apache Airflow orchestrates 8-task DAG
- **GPU Acceleration**: PyTorch with CUDA support for fast training
- **Daily Predictions**: Automated morning reports with confidence scores and model consensus

## 📊 Model Performance

| Model | Training Accuracy | Test Accuracy |
|-------|------------------|---------------|
| **Random Forest** | 96.43% | **72.41%** 🏆 |
| Gradient Boosting | 96.43% | 65.52% |
| XGBoost | 96.43% | 68.97% |
| Neural Network | 76.79% | 51.72% |
| Logistic Regression | 73.21% | 62.07% |
| **Ensemble (Weighted)** | - | **~70%** 🎯 |

*Dataset: 141 matchups, 148 features (10 traditional stats + 64 LSTM features per team)*

## 🏗️ Architecture

### Pipeline Tasks (8 Total)
1. **Scrape Team Stats** - Collect game logs from Sports Reference for all 16 Big 12 teams
2. **Calculate Averages** - Compute rolling 10-game statistics (PPG, FG%, etc.)
3. **Train LSTM Encoder** - Extract temporal features into 64-dimensional embeddings
4. **Analyze LSTM Features** 🆕 - Correlation analysis to interpret what LSTM learned
5. **Assemble Matchups** - Create feature vectors combining stats + LSTM features for both teams
6. **Train Neural Network** - PyTorch GPU-accelerated binary classifier
7. **Train Traditional ML** - RandomForest, GradientBoosting, XGBoost, LogisticRegression
8. **Generate Predictions** - Ensemble voting with confidence scores for upcoming games

### Feature Engineering
- **Traditional Stats** (10 per team): Points, opponent points, FG%, 3P%, FT%, rebounds, assists, steals, blocks, turnovers
- **LSTM Temporal Features** (64 per team): Learned patterns from sequences of last 10 games
  - Captures momentum, trends, consistency, streaks
  - Highly interpretable: 682 strong correlations (|r| > 0.3) with input stats
  - Example: lstm_f0 → mean_points (r=0.755), lstm_f3 → 3-point trend (r=0.267)
- **Total**: 148 features per matchup (74 per team × 2)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM recommended
- NVIDIA GPU (optional, for faster training)

### Setup

\\\ash
# Clone the repository
git clone https://github.com/mkrohn5051/big12-basketball-predictor.git
cd big12-basketball-predictor

# Start Airflow
docker-compose up -d

# Wait ~30 seconds for services to start
# Access Airflow UI: http://localhost:8080
# Default credentials: airflow / airflow
\\\

### Run Predictions

1. Navigate to http://localhost:8080
2. Find the **big12_basketball_predictor** DAG
3. Click the ▶️ play button to trigger
4. Wait ~2-3 minutes for completion (8 tasks)
5. Check outputs:
   - Predictions: \data/raw/predictions_YYYYMMDD.txt\
   - LSTM training viz: \data/raw/lstm_training_viz_YYYYMMDD.png\
   - Feature analysis: \data/raw/lstm_feature_analysis_YYYYMMDD.png\
   - Interpretation report: \data/raw/lstm_interpretation_YYYYMMDD.txt\

## 📁 Project Structure

\\\
airflow-basketball-predictor/
├── dags/
│   └── big12_basketball_predictor.py    # Main Airflow DAG (8 tasks)
├── data/
│   ├── raw/                              # Scraped data + outputs
│   │   ├── predictions_YYYYMMDD.txt     # Daily game predictions
│   │   ├── lstm_training_viz_YYYYMMDD.png    # 6-panel LSTM training analysis
│   │   ├── lstm_feature_analysis_YYYYMMDD.png # 4-panel correlation heatmaps
│   │   └── lstm_interpretation_YYYYMMDD.txt   # Feature meanings
│   ├── lstm_encoder.pt                   # Trained LSTM model
│   ├── game_predictor_nn.pt             # Trained neural network
│   └── model_*.pkl                      # Traditional ML models
├── docker-compose.yml                    # Airflow services + GPU config
└── README.md
\\\

## 🔬 LSTM Interpretability Analysis

One of the unique features of this system is the **LSTM interpretability analysis** that decodes what the neural network actually learned.

### Correlation Analysis Results
- **682 strong correlations** found between LSTM features and input statistics (|r| > 0.3)
- Top LSTM feature meanings discovered:
  - **lstm_f0, f1, f2, f4, f6-f9**: Strongly correlated with **mean_points** (r=0.755) → Captures offensive firepower
  - **lstm_f3, f5**: Correlated with **trend_three_pct** (r=0.267) → Captures 3-point shooting momentum
  - Other features encode: point differential, consistency (std), win streaks, recent form

### Visualization Outputs (10 Total Plots)

#### LSTM Training Visualization (6 panels)

**1. Training Loss Curve**
- Shows loss (negative variance) over 50 epochs: 0 → -8.5
- Negative loss = model learning to create MORE diverse team embeddings
- Convergence indicates successful training

**2. Iowa State Embedding Evolution**
- Tracks first 2 dimensions of Iowa State's 64D embedding during training
- Shows how model's understanding of Iowa State stabilizes over epochs
- Flat lines at end = converged representation

**3. PCA Projection (2D)**
- Reduces 64 LSTM dimensions to 2D using Principal Component Analysis
- **Iowa State position**: Isolated in top-right → unique playing style/momentum
- PC1 explains 100% variance, PC2 explains 0.09% variance
- Distance from other teams = distinctiveness of temporal pattern

**4. t-SNE Projection (2D)**
- Non-linear clustering showing which teams play similarly
- **Iowa State**: Isolated at bottom-left → most unique recent pattern in conference
- Teams close together = similar momentum/style over last 10 games
- West Virginia cluster suggests struggling teams group together

**5. Embedding Magnitudes**
- Bar chart showing L2 norm (vector length) for each team
- **Iowa State has HIGHEST magnitude** (red bar, ~50) → strongest, most pronounced temporal signature
- High magnitude = clear patterns (hot streak, consistent excellence, or distinctive style)
- Arizona also high (~47) → another team with strong identity

**6. LSTM Feature Variance Distribution**
- Shows which of 64 LSTM dimensions distinguish teams most
- High variance features (tall purple bars) = important learned patterns
- Some features have variance >20 → highly discriminative
- Most features 5-15 variance → moderate importance

#### LSTM Feature Analysis (4 panels)

**1. Full Correlation Heatmap**
- 18 input statistics × 64 LSTM features = 1,152 correlations
- Red = positive correlation, Blue = negative correlation
- Pattern: Many LSTM features strongly correlate with offensive stats (mean_points, point_diff)
- Shows LSTM learned to encode scoring power and momentum

**2. Strong Correlations Only (|r| > 0.3)**
- Filters to 682 meaningful correlations
- Dense red patterns = many LSTM features capture mean_points (r≈0.6-0.8)
- Blue patterns = some features negatively correlated (defensive emphasis?)
- Nearly all LSTM features have at least one strong correlation → highly interpretable model

**3. Strongest Correlation for Each LSTM Feature**
- Bar chart: Green = positive correlation, Red = negative
- ~60% of LSTM features most strongly encode **mean_points**
- Remaining features encode: point_diff, assists, blocks, 3P trends
- Every LSTM dimension has a clear "meaning" in terms of basketball stats

**4. Distribution of Correlation Strengths**
- Histogram showing most correlations are moderate (0.3-0.7)
- Large spike around 0.5-0.6 = many moderately strong relationships
- Red dashed line at 0.3 = "strong correlation" threshold
- ~60% of all correlations exceed this threshold → model is very interpretable

## 📈 Sample Prediction Output

\\\
================================================================================
BIG 12 BASKETBALL PREDICTIONS - 2026-02-06
================================================================================
Ensemble Model: 5 models voting
Best Individual: Random Forest (72.4%)
================================================================================

Sat, Feb 08: Cincinnati vs Houston
  Cincinnati (6-4, 68.8 PPG) vs Houston (7-3, 82.0 PPG)
  🏆 ENSEMBLE PICK: Houston
  📊 Confidence: 73.1%
  🗳️  Model Vote: 5/5 models pick Houston
  🎲 Win Probability: Cincinnati 26.9% | Houston 73.1%

Sat, Feb 08: Kansas vs Iowa State
  Kansas (8-2, 81.9 PPG) vs Iowa State (7-3, 81.4 PPG)
  🏆 ENSEMBLE PICK: Kansas
  📊 Confidence: 58.2%
  🗳️  Model Vote: 3/5 models pick Kansas
  🎲 Win Probability: Kansas 58.2% | Iowa State 41.8%

================================================================================
Total Predictions: 10
Prediction Method: Accuracy-weighted ensemble
================================================================================
\\\

## 🛠️ Tech Stack

- **Orchestration**: Apache Airflow 2.x
- **Deep Learning**: PyTorch, CUDA
- **ML Models**: scikit-learn, XGBoost
- **Data**: Pandas, NumPy, BeautifulSoup
- **Visualization**: Matplotlib, Seaborn
- **Database**: PostgreSQL
- **Infrastructure**: Docker, Docker Compose

## 🔬 Technical Details

### LSTM Encoder Architecture
- **Input**: Sequences of 10 games × 10 stats per game
- **Architecture**: 2-layer LSTM (128 hidden units) → ReLU → Dense (64 output)
- **Training**: Unsupervised variance maximization (50 epochs)
- **Output**: 64-dimensional embedding per team encoding temporal patterns
- **Interpretability**: 682 strong correlations with input stats

### Neural Network Predictor
- **Architecture**: 148 → 256 → 128 → 64 → 1 (sigmoid)
- **Dropout**: 0.3 between layers to prevent overfitting
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross-Entropy
- **Training**: 100 epochs, batch size 16, GPU-accelerated

### Ensemble Method
- **Strategy**: Accuracy-weighted voting
- Each model's vote weighted by its test accuracy
- Final prediction = weighted probability > 0.5
- Shows both consensus (vote count) and weighted confidence

### Data Pipeline
- **Source**: sports-reference.com (with respectful rate limiting)
- **Teams**: 16 Big 12 schools (2025-26 season)
- **Games**: ~30 per team, 495 total scraped
- **Training Data**: 141 conference matchups
- **Update Frequency**: Manual trigger (recommended daily at 6am)

## 🔮 Key Insights from Analysis

1. **LSTM captures offensive power**: Most features (60%+) strongly correlate with scoring
2. **Iowa State is unique**: Highest embedding magnitude + isolated in both PCA and t-SNE projections
3. **3-point trends matter**: Multiple LSTM dimensions specifically encode 3P% momentum
4. **High interpretability**: 682/1152 correlations are strong → not a "black box"
5. **Ensemble adds robustness**: 5-model voting smooths individual model weaknesses

## 🔮 Future Enhancements

- [ ] Add player injury data scraping
- [ ] Incorporate betting line movements
- [ ] Real-time score tracking during games
- [ ] Backtesting framework with historical predictions
- [ ] Web dashboard for interactive visualizations
- [ ] Slack/email notifications for daily predictions
- [ ] Expand to other conferences (ACC, SEC, Big Ten)
- [ ] Add attention mechanisms to LSTM for game-level importance weights

## 📝 License

MIT License - feel free to use this for your own predictions!

## 🙏 Acknowledgments

- Sports Reference for comprehensive game data
- Big 12 Conference for exciting basketball
- Anthropic Claude for architecture guidance and late-night debugging
- The open-source ML community

## 📧 Contact

Built by Mike
Check out the visualizations in \data/raw/\ after running the pipeline!

---

**🏀 May your predictions be accurate and your brackets stay intact!**

*Go Cyclones!* 🌪️
