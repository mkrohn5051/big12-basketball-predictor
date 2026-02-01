# 🏀 Big 12 Basketball Game Predictor

An end-to-end machine learning pipeline that predicts Big 12 basketball game outcomes using LSTM temporal features and ensemble models.

## 🎯 Overview

This system scrapes live game data from Sports Reference, trains multiple ML models, and generates daily predictions for upcoming Big 12 basketball games. Built with Apache Airflow for orchestration and PyTorch for GPU-accelerated deep learning.

## ⚡ Features

- **Automated Data Collection**: Web scraper for 16 Big 12 teams with respectful rate limiting
- **Temporal Feature Engineering**: LSTM encoder captures momentum and trends from last 10 games
- **Multiple ML Models**: Neural network, Random Forest, Gradient Boosting, XGBoost
- **Production Pipeline**: Apache Airflow orchestrates 7-task DAG
- **GPU Acceleration**: PyTorch with CUDA support for fast training
- **Daily Predictions**: Automated morning reports with confidence scores

## 📊 Model Performance

| Model | Training Accuracy | Test Accuracy |
|-------|------------------|---------------|
| **Gradient Boosting** | 96.43% | **72.41%** 🏆 |
| Random Forest | 96.43% | 68.97% |
| XGBoost | 96.43% | 68.97% |
| Neural Network | 76.79% | 65.52% |
| Logistic Regression | 73.21% | 58.62% |

*Dataset: 141 matchups, 148 features (traditional stats + 64 LSTM features per team)*

## 🏗️ Architecture

### Pipeline Tasks
1. **Scrape Team Stats** - Collect game logs from Sports Reference
2. **Calculate Averages** - Compute rolling 10-game statistics
3. **Train LSTM Encoder** - Extract temporal features (64-dim embeddings)
4. **Assemble Matchups** - Create feature vectors for training
5. **Train Neural Network** - PyTorch GPU-accelerated model
6. **Train Traditional ML** - RandomForest, GradientBoosting, XGBoost
7. **Generate Predictions** - Daily game predictions with confidence

### Feature Engineering
- **Traditional Stats** (10 per team): PPG, FG%, 3P%, FT%, rebounds, assists, steals, blocks, turnovers
- **LSTM Features** (64 per team): Temporal patterns from last 10 games
- **Total**: 148 features per matchup (74 per team × 2)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM recommended
- NVIDIA GPU (optional, for faster training)

### Setup

\\\ash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/big12-basketball-predictor.git
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
4. Wait ~2-3 minutes for completion
5. Check predictions: \data/raw/predictions_YYYYMMDD.txt\

## 📁 Project Structure

\\\
airflow-basketball-predictor/
├── dags/
│   └── big12_basketball_predictor.py    # Main Airflow DAG
├── data/
│   ├── raw/                              # Scraped data + prediction reports
│   ├── lstm_encoder.pt                   # Trained LSTM model
│   └── game_predictor_nn.pt             # Trained neural network
├── docker-compose.yaml                   # Airflow services
└── README.md
\\\

## 🔬 Technical Details

### LSTM Encoder
- **Architecture**: 2-layer LSTM (128 hidden units) → Dense (64 output)
- **Input**: 10 games × 10 stats
- **Output**: 64-dimensional embedding per team
- **Training**: Unsupervised, maximizes feature variance

### Neural Network Predictor
- **Architecture**: 148 → 256 → 128 → 64 → 1 (sigmoid)
- **Dropout**: 0.3 between layers
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross-Entropy
- **Training**: 100 epochs, batch size 16

### Data Pipeline
- **Source**: sports-reference.com
- **Teams**: 16 Big 12 schools (2025-26 season)
- **Games**: ~30 per team (495 total)
- **Update Frequency**: Manual trigger (recommended daily)

## 📈 Sample Output

\\\
======================================================================
BIG 12 BASKETBALL PREDICTIONS - 2026-02-01
======================================================================

Sat, Jan 31: Cincinnati vs Houston
  → Predicted Winner: Houston
  → Confidence: 73.1%

Sat, Jan 31: UCF vs Texas Tech
  → Predicted Winner: Texas Tech
  → Confidence: 63.3%

Sat, Jan 31: West Virginia vs Baylor
  → Predicted Winner: Baylor
  → Confidence: 59.2%
...
======================================================================
\\\

## 🛠️ Tech Stack

- **Orchestration**: Apache Airflow 2.x
- **Deep Learning**: PyTorch, CUDA
- **ML Models**: scikit-learn, XGBoost
- **Data**: Pandas, NumPy, BeautifulSoup
- **Database**: PostgreSQL
- **Infrastructure**: Docker, Docker Compose

## 🔮 Future Enhancements

- [ ] Add player injury data
- [ ] Incorporate betting line movements
- [ ] Real-time score tracking
- [ ] Model ensemble voting
- [ ] Backtesting framework
- [ ] Web dashboard for predictions
- [ ] Slack/email notifications

## 📝 License

MIT License - feel free to use this for your own predictions!

## 🙏 Acknowledgments

- Sports Reference for game data
- Big 12 Conference for amazing basketball
- Claude for late-night debugging sessions

## 📧 Contact

Built by Mike - Engineering Manager & ML Enthusiast
\\\

---

**🏀 May your predictions be accurate and your brackets be busted in the best way possible!**
