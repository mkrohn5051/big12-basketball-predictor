from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# LSTM bits
LSTM_FEATURE_DIM = 64  # OUTPUT DIM FOR LSTM FEAT
SEQUENCE_LENGTH = 10  # NUM GAMES TO COVER

# Configuration
CURRENT_SEASON = 2026
RAW_DIR = Path('/opt/airflow/data/raw')
DB_CONN = 'postgresql+psycopg2://airflow:airflow@postgres/airflow'

# Big 12 Teams (2025-26 season)
BIG12_TEAMS = {
    'Iowa State': 'iowa-state',
    'Brigham Young': 'brigham-young',
    'Arizona': 'arizona',
    'Arizona State': 'arizona-state',
    'Kansas': 'kansas',
    'Kansas State': 'kansas-state',
    'Texas Tech': 'texas-tech',
    'Colorado': 'colorado',
    'Cincinnati': 'cincinnati',
    'Oklahoma State': 'oklahoma-state',
    'Houston': 'houston',
    'Baylor': 'baylor',
    'TCU': 'texas-christian',
    'Utah': 'utah',
    'West Virginia': 'west-virginia',
    'UCF': 'central-florida',
}

# Default args for the DAG
default_args = {
    'owner': 'mike',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'big12_basketball_predictor',
    default_args=default_args,
    description='Predict Big 12 basketball games using LSTM + stats',
    schedule_interval=None,  # Manual trigger
    catchup=False,
)

def scrape_team_stats(**context):
    """Task 1: Scrape all Big 12 team game logs"""
    print("Starting Big 12 game log scraper...")
    
    all_games = []
    
    for team_name, team_slug in BIG12_TEAMS.items():
        print(f"\nScraping {team_name}...")
        
        url = f"https://www.sports-reference.com/cbb/schools/{team_slug}/men/{CURRENT_SEASON}-gamelogs.html"
        
        try:
            # Respectful scraping - wait between requests
            time.sleep(random.uniform(2, 5))
            
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the game log table
            table = soup.find('table', {'id': 'team_game_log'})
            
            if not table:
                print(f"WARNING: Could not find game log table for {team_name}")
                continue
            
            # Parse table rows
            rows = table.find('tbody').find_all('tr')
            
            for row in rows:
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                
                cells = row.find_all(['th', 'td'])
                
                if len(cells) < 10:
                    continue
                
                # Extract game data - CORRECTED INDICES
                # cells[0] = rank (th), cells[1] = game_num, cells[2] = date, etc.
                game_data = {
                    'team': team_name,
                    'team_slug': team_slug,
                    'game_num': cells[1].text.strip(),
                    'date': cells[2].text.strip(),
                    'location': cells[3].text.strip(),
                    'opponent': cells[4].text.strip(),
                    'game_type': cells[5].text.strip(),
                    'result': cells[6].text.strip(),
                    'team_score': cells[7].text.strip(),
                    'opp_score': cells[8].text.strip(),
                    'overtimes': cells[9].text.strip() if len(cells) > 9 else '',
                    'fg': cells[10].text.strip() if len(cells) > 10 else '',
                    'fga': cells[11].text.strip() if len(cells) > 11 else '',
                    'fg_pct': cells[12].text.strip() if len(cells) > 12 else '',
                    'threes': cells[13].text.strip() if len(cells) > 13 else '',
                    'threes_att': cells[14].text.strip() if len(cells) > 14 else '',
                    'three_pct': cells[15].text.strip() if len(cells) > 15 else '',
                    'ft': cells[19].text.strip() if len(cells) > 19 else '',
                    'fta': cells[20].text.strip() if len(cells) > 20 else '',
                    'ft_pct': cells[21].text.strip() if len(cells) > 21 else '',
                    'orb': cells[22].text.strip() if len(cells) > 22 else '',
                    'drb': cells[23].text.strip() if len(cells) > 23 else '',
                    'trb': cells[24].text.strip() if len(cells) > 24 else '',
                    'ast': cells[25].text.strip() if len(cells) > 25 else '',
                    'stl': cells[26].text.strip() if len(cells) > 26 else '',
                    'blk': cells[27].text.strip() if len(cells) > 27 else '',
                    'tov': cells[28].text.strip() if len(cells) > 28 else '',
                    'pf': cells[29].text.strip() if len(cells) > 29 else '',
                }
                
                all_games.append(game_data)
            
            print(f"  ✓ Scraped {len([g for g in all_games if g['team'] == team_name])} games for {team_name}")
            
        except Exception as e:
            print(f"ERROR scraping {team_name}: {e}")
            continue
    
    # Save to CSV
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_games)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RAW_DIR / f'big12_games_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Scraped {len(all_games)} total games")
    print(f"✓ Saved to: {csv_path}")
    
    # Save to Postgres
    engine = create_engine(DB_CONN)
    df.to_sql('game_logs', engine, if_exists='replace', index=False)
    
    print("✓ Saved to Postgres table: game_logs")
    
    context['ti'].xcom_push(key='total_games', value=len(all_games))
    context['ti'].xcom_push(key='csv_path', value=str(csv_path))

def calculate_team_averages(**context):
    """Task 2: Calculate rolling 10-game averages for each team"""
    print("Calculating team rolling averages...")
    
    engine = create_engine(DB_CONN)
    
    # Read all games, convert numeric columns
    df = pd.read_sql('SELECT * FROM game_logs', engine)
    print(f"Loaded {len(df)} total rows from database")
    
    # Filter only completed games (where result is not empty)
    df = df[df['result'].notna() & (df['result'] != '')]
    print(f"After filtering for completed games: {len(df)} rows")
    
    # Convert numeric columns
    numeric_cols = ['team_score', 'opp_score', 'fg', 'fga', 'fg_pct', 'threes', 'threes_att', 
                    'three_pct', 'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb', 'ast', 'stl', 
                    'blk', 'tov', 'pf']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse dates - coerce errors to NaT (Not a Time)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"After date parsing, rows with valid dates: {df['date'].notna().sum()}")
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])
    print(f"After dropping invalid dates: {len(df)} rows")
    
    # Sort by team and date
    df = df.sort_values(['team', 'date'])
    
    # Calculate rolling 10-game averages for each team
    team_stats = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        
        # Take last 10 games (or all if fewer than 10)
        recent_games = team_df.tail(10)
        
        if len(recent_games) > 0:
            stats = {
                'team': team,
                'games_played': len(recent_games),
                'avg_points': recent_games['team_score'].mean(),
                'avg_opp_points': recent_games['opp_score'].mean(),
                'avg_fg_pct': recent_games['fg_pct'].mean(),
                'avg_three_pct': recent_games['three_pct'].mean(),
                'avg_ft_pct': recent_games['ft_pct'].mean(),
                'avg_rebounds': recent_games['trb'].mean(),
                'avg_assists': recent_games['ast'].mean(),
                'avg_steals': recent_games['stl'].mean(),
                'avg_blocks': recent_games['blk'].mean(),
                'avg_turnovers': recent_games['tov'].mean(),
                'wins': len(recent_games[recent_games['result'] == 'W']),
                'losses': len(recent_games[recent_games['result'] == 'L']),
            }
            team_stats.append(stats)
            print(f"{team}: {stats['games_played']} games, {stats['avg_points']:.1f} PPG")
    
    # Save to database
    stats_df = pd.DataFrame(team_stats)
    stats_df.to_sql('team_current_stats', engine, if_exists='replace', index=False)
    
    print(f"\n✓ Calculated averages for {len(team_stats)} teams")
    print(f"✓ Saved to Postgres table: team_current_stats")
    
    context['ti'].xcom_push(key='teams_processed', value=len(team_stats))

class LSTMEncoder(nn.Module):
    """LSTM network to encode team game sequences into feature vectors"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=64):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Project LSTM output to fixed dimension
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # Shape: (batch, hidden_dim)
        
        # Project to output dimension
        out = self.fc(last_hidden)
        out = self.relu(out)
        
        return out

def train_lstm_encoder(**context):
    """Task 3: Train LSTM to encode game sequences"""
    print("Training LSTM encoder...")
    
    engine = create_engine(DB_CONN)
    
    # Load completed games
    df = pd.read_sql('SELECT * FROM game_logs', engine)
    df = df[df['result'].notna() & (df['result'] != '')]
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['team', 'date'])
    
    # Select features for LSTM
    feature_cols = ['team_score', 'opp_score', 'fg_pct', 'three_pct', 'ft_pct', 
                    'trb', 'ast', 'stl', 'blk', 'tov']
    
    # Convert to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Prepare sequences for each team
    sequences = []
    team_names = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        team_df = team_df.dropna(subset=feature_cols)
        
        if len(team_df) >= SEQUENCE_LENGTH:
            # Take last SEQUENCE_LENGTH games
            recent = team_df.tail(SEQUENCE_LENGTH)
            seq = recent[feature_cols].values
            sequences.append(seq)
            team_names.append(team)
    
    if len(sequences) == 0:
        print("ERROR: No valid sequences found!")
        return
    
    # Convert to numpy array and normalize
    X = np.array(sequences, dtype=np.float32)
    print(f"Prepared {len(sequences)} sequences, shape: {X.shape}")
    
    # Normalize each feature across all sequences
    means = X.mean(axis=(0, 1), keepdims=True)
    stds = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_normalized = (X - means) / stds
    
    # Save normalization params
    norm_params = pd.DataFrame({
        'feature': feature_cols,
        'mean': means.squeeze(),
        'std': stds.squeeze()
    })
    norm_params.to_sql('lstm_norm_params', engine, if_exists='replace', index=False)
    print("✓ Saved normalization parameters")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_normalized)
    
    # Initialize model
    input_dim = len(feature_cols)
    model = LSTMEncoder(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=LSTM_FEATURE_DIM
    )
    
    print(f"Model architecture: {input_dim} -> LSTM(128) -> {LSTM_FEATURE_DIM}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Track training history
    training_history = {
        'epoch': [],
        'loss': [],
        'iowa_state_embedding': []
    }
    
    iowa_state_idx = team_names.index('Iowa State') if 'Iowa State' in team_names else None
    
    # Training loop with history tracking
    model.train()
    epochs = 50
    
    print("\nTraining LSTM...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        encoded = model(X_tensor)
        
        # Loss: maximize variance across samples (encourage diverse embeddings)
        loss = torch.var(encoded, dim=0).mean()
        loss = -loss  # We want high variance
        
        loss.backward()
        optimizer.step()
        
        # Track history
        training_history['epoch'].append(epoch + 1)
        training_history['loss'].append(loss.item())
        
        # Track Iowa State embedding if available
        if iowa_state_idx is not None:
            iowa_embedding = encoded[iowa_state_idx].detach().numpy()
            training_history['iowa_state_embedding'].append(iowa_embedding.copy())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    print("✓ LSTM training complete")
    
    # Generate final features
    model.eval()
    with torch.no_grad():
        lstm_features = model(X_tensor).numpy()
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(lstm_features, columns=[f'lstm_f{i}' for i in range(LSTM_FEATURE_DIM)])
    feature_df['team'] = team_names
    
    # Save to database
    feature_df.to_sql('lstm_features', engine, if_exists='replace', index=False)
    print(f"✓ Saved LSTM features for {len(team_names)} teams")
    
    # === VISUALIZATION ===
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("\nGenerating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training Loss Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(training_history['epoch'], training_history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Negative Variance)', fontsize=12)
    ax1.set_title('LSTM Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Iowa State Embedding Evolution (first 2 dimensions)
    if iowa_state_idx is not None:
        ax2 = plt.subplot(2, 3, 2)
        iowa_embeddings = np.array(training_history['iowa_state_embedding'])
        ax2.plot(training_history['epoch'], iowa_embeddings[:, 0], 'r-', label='Dim 0', linewidth=2)
        ax2.plot(training_history['epoch'], iowa_embeddings[:, 1], 'g-', label='Dim 1', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Embedding Value', fontsize=12)
        ax2.set_title('Iowa State Embedding Evolution (First 2 Dims)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. PCA Projection (2D)
    ax3 = plt.subplot(2, 3, 3)
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(lstm_features)
    
    # Plot all teams
    ax3.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], 
                c='lightblue', s=100, alpha=0.6, edgecolors='black')
    
    # Highlight Iowa State
    if iowa_state_idx is not None:
        ax3.scatter(embeddings_2d_pca[iowa_state_idx, 0], 
                   embeddings_2d_pca[iowa_state_idx, 1],
                   c='red', s=300, marker='*', edgecolors='darkred', linewidth=2,
                   label='Iowa State', zorder=5)
    
    # Label all teams
    for i, team in enumerate(team_names):
        ax3.annotate(team, (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]),
                    fontsize=8, alpha=0.7, ha='center')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax3.set_title('Team Embeddings (PCA Projection)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. t-SNE Projection (2D)
    ax4 = plt.subplot(2, 3, 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(lstm_features)-1))
    embeddings_2d_tsne = tsne.fit_transform(lstm_features)
    
    ax4.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                c='lightgreen', s=100, alpha=0.6, edgecolors='black')
    
    if iowa_state_idx is not None:
        ax4.scatter(embeddings_2d_tsne[iowa_state_idx, 0],
                   embeddings_2d_tsne[iowa_state_idx, 1],
                   c='red', s=300, marker='*', edgecolors='darkred', linewidth=2,
                   label='Iowa State', zorder=5)
    
    for i, team in enumerate(team_names):
        ax4.annotate(team, (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
                    fontsize=8, alpha=0.7, ha='center')
    
    ax4.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax4.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax4.set_title('Team Embeddings (t-SNE Projection)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Embedding Magnitude by Team
    ax5 = plt.subplot(2, 3, 5)
    embedding_norms = np.linalg.norm(lstm_features, axis=1)
    sorted_idx = np.argsort(embedding_norms)[::-1]
    
    colors = ['red' if team_names[i] == 'Iowa State' else 'skyblue' for i in sorted_idx]
    ax5.barh(range(len(team_names)), embedding_norms[sorted_idx], color=colors)
    ax5.set_yticks(range(len(team_names)))
    ax5.set_yticklabels([team_names[i] for i in sorted_idx], fontsize=9)
    ax5.set_xlabel('Embedding Magnitude', fontsize=12)
    ax5.set_title('Team Embedding Magnitudes', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Feature Variance
    ax6 = plt.subplot(2, 3, 6)
    feature_variance = np.var(lstm_features, axis=0)
    ax6.bar(range(len(feature_variance)), feature_variance, color='purple', alpha=0.7)
    ax6.set_xlabel('LSTM Feature Index', fontsize=12)
    ax6.set_ylabel('Variance', fontsize=12)
    ax6.set_title('LSTM Feature Variance Distribution', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = Path('/opt/airflow/data/raw') / f'lstm_training_viz_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {viz_path}")
    
    # Save model
    model_path = Path('/opt/airflow/data') / 'lstm_encoder.pt'
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'feature_cols': feature_cols,
        'lstm_dim': LSTM_FEATURE_DIM,
        'training_history': training_history
    }, model_path)
    print(f"✓ Saved model to {model_path}")
    
    context['ti'].xcom_push(key='lstm_features_generated', value=len(team_names))

def analyze_lstm_features(**context):
    """Task 3b: Analyze what LSTM features represent"""
    print("Analyzing LSTM feature interpretability...")
    
    engine = create_engine(DB_CONN)
    
    # Load the data
    df = pd.read_sql('SELECT * FROM game_logs', engine)
    df = df[df['result'].notna() & (df['result'] != '')]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['team', 'date'])
    
    # Original input features
    feature_cols = ['team_score', 'opp_score', 'fg_pct', 'three_pct', 'ft_pct', 
                    'trb', 'ast', 'stl', 'blk', 'tov']
    
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Load LSTM features and normalization
    lstm_features_df = pd.read_sql('SELECT * FROM lstm_features', engine)
    norm_params = pd.read_sql('SELECT * FROM lstm_norm_params', engine)
    
    # Calculate rolling statistics for each team (what the LSTM "sees")
    team_rolling_stats = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        team_df = team_df.dropna(subset=feature_cols)
        
        if len(team_df) >= SEQUENCE_LENGTH:
            # Get last 10 games
            recent = team_df.tail(SEQUENCE_LENGTH)
            
            # Calculate various statistics on the sequence
            stats = {
                'team': team,
                # Means
                'mean_points': recent['team_score'].mean(),
                'mean_opp_points': recent['opp_score'].mean(),
                'mean_fg_pct': recent['fg_pct'].mean(),
                'mean_three_pct': recent['three_pct'].mean(),
                'mean_ft_pct': recent['ft_pct'].mean(),
                'mean_rebounds': recent['trb'].mean(),
                'mean_assists': recent['ast'].mean(),
                'mean_steals': recent['stl'].mean(),
                'mean_blocks': recent['blk'].mean(),
                'mean_turnovers': recent['tov'].mean(),
                # Trends (linear regression slope)
                'trend_points': np.polyfit(range(len(recent)), recent['team_score'], 1)[0],
                'trend_fg_pct': np.polyfit(range(len(recent)), recent['fg_pct'].fillna(0), 1)[0],
                'trend_three_pct': np.polyfit(range(len(recent)), recent['three_pct'].fillna(0), 1)[0],
                # Volatility (std dev)
                'std_points': recent['team_score'].std(),
                'std_fg_pct': recent['fg_pct'].std(),
                # Win streak
                'wins_last_10': (recent['result'] == 'W').sum(),
                # Recent performance (last 3 games avg)
                'recent_3_points': recent.tail(3)['team_score'].mean(),
                # Point differential
                'avg_point_diff': (recent['team_score'] - recent['opp_score']).mean(),
            }
            team_rolling_stats.append(stats)
    
    rolling_df = pd.DataFrame(team_rolling_stats)
    
    # Merge with LSTM features
    combined_df = rolling_df.merge(lstm_features_df, on='team', how='inner')
    
    print(f"Analyzing {len(combined_df)} teams")
    
    # Calculate correlations
    stat_cols = [col for col in rolling_df.columns if col != 'team']
    lstm_cols = [col for col in lstm_features_df.columns if col.startswith('lstm_f')]
    
    # Correlation matrix: each stat vs each LSTM feature
    correlations = []
    
    for stat in stat_cols:
        for lstm_feat in lstm_cols:
            corr = combined_df[stat].corr(combined_df[lstm_feat])
            correlations.append({
                'stat': stat,
                'lstm_feature': lstm_feat,
                'correlation': corr
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Drop NaN correlations
    corr_df = corr_df.dropna(subset=['correlation'])
    
    # Save to database
    corr_df.to_sql('lstm_feature_correlations', engine, if_exists='replace', index=False)
    
    if len(corr_df) == 0:
        print("WARNING: No valid correlations found. Skipping analysis.")
        context['ti'].xcom_push(key='features_analyzed', value=0)
        return
    
    # Find top correlations for each LSTM feature
    print("\nTop correlations for each LSTM feature:")
    lstm_feature_meanings = []
    
    for lstm_feat in lstm_cols[:10]:  # Show first 10 features
        feat_corrs = corr_df[corr_df['lstm_feature'] == lstm_feat]
        if len(feat_corrs) > 0:
            top_corrs = feat_corrs.nlargest(3, 'correlation', keep='all')
            print(f"\n{lstm_feat}:")
            for _, row in top_corrs.iterrows():
                print(f"  {row['stat']}: {row['correlation']:.3f}")
            
            # Store interpretation
            if len(top_corrs) > 0:
                best_stat = top_corrs.iloc[0]['stat']
                best_corr = top_corrs.iloc[0]['correlation']
                lstm_feature_meanings.append({
                    'lstm_feature': lstm_feat,
                    'best_correlation': best_stat,
                    'correlation_value': best_corr
                })
    
    # === VISUALIZATIONS ===
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nGenerating correlation visualizations...")
    
    # Create pivot table for heatmap
    corr_pivot = corr_df.pivot(index='stat', columns='lstm_feature', values='correlation')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Full correlation heatmap
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(corr_pivot, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'}, ax=ax1)
    ax1.set_title('Input Stats vs LSTM Features Correlation Heatmap', fontsize=16, fontweight='bold')
    ax1.set_xlabel('LSTM Feature Index', fontsize=12)
    ax1.set_ylabel('Input Statistics', fontsize=12)
    
    # 2. Strongest correlations only (abs > 0.3)
    ax2 = plt.subplot(2, 2, 2)
    strong_corr = corr_df[abs(corr_df['correlation']) > 0.3].copy()
    
    if len(strong_corr) > 0:
        strong_pivot = strong_corr.pivot(index='stat', columns='lstm_feature', values='correlation')
        sns.heatmap(strong_pivot, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    cbar_kws={'label': 'Correlation'}, ax=ax2, annot=True, fmt='.2f')
        ax2.set_title('Strong Correlations Only (|r| > 0.3)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('LSTM Feature Index', fontsize=12)
        ax2.set_ylabel('Input Statistics', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No strong correlations found (|r| > 0.3)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Strong Correlations Only (|r| > 0.3)', fontsize=16, fontweight='bold')
    
    # 3. Top correlated stat for each LSTM feature
    ax3 = plt.subplot(2, 2, 3)
    
    # Find strongest correlation for each LSTM feature (by absolute value)
    def get_max_abs_idx(group):
        abs_corr = group['correlation'].abs()
        if len(abs_corr) > 0:
            return abs_corr.idxmax()
        return None
    
    max_indices = corr_df.groupby('lstm_feature').apply(get_max_abs_idx).dropna()
    top_corrs_per_lstm = corr_df.loc[max_indices].copy()
    top_corrs_per_lstm = top_corrs_per_lstm.sort_values('correlation', ascending=True)
    
    if len(top_corrs_per_lstm) > 0:
        colors = ['red' if x < 0 else 'green' for x in top_corrs_per_lstm['correlation']]
        ax3.barh(range(len(top_corrs_per_lstm)), top_corrs_per_lstm['correlation'], color=colors, alpha=0.6)
        ax3.set_yticks(range(len(top_corrs_per_lstm)))
        ax3.set_yticklabels([f"{row['lstm_feature']}: {row['stat']}" 
                             for _, row in top_corrs_per_lstm.iterrows()], fontsize=7)
        ax3.set_xlabel('Correlation Coefficient', fontsize=12)
        ax3.set_title('Strongest Correlation for Each LSTM Feature', fontsize=16, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'No correlations to display', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
    
    # 4. Distribution of correlation strengths
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(corr_df['correlation'].abs(), bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Strong correlation threshold')
    ax4.set_xlabel('Absolute Correlation Value', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Correlation Strengths', fontsize=16, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = Path('/opt/airflow/data/raw') / f'lstm_feature_analysis_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analysis visualization saved to {viz_path}")
    
    # Create interpretation report
    report = []
    report.append("="*80)
    report.append("LSTM FEATURE INTERPRETATION ANALYSIS")
    report.append("="*80)
    report.append("")
    report.append("This analysis correlates LSTM features with interpretable input statistics")
    report.append("to understand what temporal patterns each LSTM dimension captures.")
    report.append("")
    report.append("="*80)
    report.append("TOP LSTM FEATURE INTERPRETATIONS")
    report.append("="*80)
    report.append("")
    
    for meaning in lstm_feature_meanings:
        report.append(f"{meaning['lstm_feature']}: Most correlated with '{meaning['best_correlation']}'")
        report.append(f"  Correlation: {meaning['correlation_value']:.3f}")
        report.append("")
    
    report.append("="*80)
    report.append(f"Total LSTM Features: {len(lstm_cols)}")
    report.append(f"Input Statistics Analyzed: {len(stat_cols)}")
    
    if len(strong_corr) > 0:
        report.append(f"Strong Correlations (|r| > 0.3): {len(strong_corr)}")
    else:
        report.append("Strong Correlations (|r| > 0.3): 0")
    
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = Path('/opt/airflow/data/raw') / f'lstm_interpretation_{datetime.now().strftime("%Y%m%d")}.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Interpretation report saved to {report_path}")
    
    context['ti'].xcom_push(key='features_analyzed', value=len(lstm_cols))

def assemble_matchup_features(**context):
    """Task 4: Create matchup feature vectors from team stats + LSTM features"""
    print("Assembling matchup features...")
    
    engine = create_engine(DB_CONN)
    
    # Load team stats and LSTM features
    team_stats = pd.read_sql('SELECT * FROM team_current_stats', engine)
    lstm_features = pd.read_sql('SELECT * FROM lstm_features', engine)
    
    # Merge stats with LSTM features
    team_data = team_stats.merge(lstm_features, on='team', how='inner')
    print(f"Loaded data for {len(team_data)} teams")
    
    # Load completed games for training labels
    games_df = pd.read_sql('SELECT * FROM game_logs', engine)
    games_df = games_df[games_df['result'].notna() & (games_df['result'] != '')]
    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')
    games_df = games_df.dropna(subset=['date'])
    
    print(f"Loaded {len(games_df)} completed games for training")
    
    # Prepare feature columns
    stat_cols = ['avg_points', 'avg_opp_points', 'avg_fg_pct', 'avg_three_pct', 
                 'avg_ft_pct', 'avg_rebounds', 'avg_assists', 'avg_steals', 
                 'avg_blocks', 'avg_turnovers']
    
    lstm_cols = [f'lstm_f{i}' for i in range(LSTM_FEATURE_DIM)]
    
    # Create matchup dataset
    matchups = []
    
    for idx, game in games_df.iterrows():
        team_a = game['team']
        team_b = game['opponent']
        result = game['result']  # 'W' or 'L'
        
        # Get team A data
        team_a_data = team_data[team_data['team'] == team_a]
        # Get team B data (need to look up full name from opponent abbreviation)
        # For now, skip if we can't find team B in our data
        if len(team_a_data) == 0:
            continue
        
        team_a_stats = team_a_data[stat_cols].values.flatten()
        team_a_lstm = team_a_data[lstm_cols].values.flatten()
        
        # Try to find team B - this is tricky because opponent is abbreviated
        # For Big 12 conference games, we can match by looking up the opponent
        team_b_candidates = team_data[team_data['team'].str.contains(team_b.split()[0], case=False, na=False)]
        
        if len(team_b_candidates) == 0:
            # Non-conference game or can't find opponent
            continue
        
        team_b_data = team_b_candidates.iloc[0]
        team_b_stats = team_b_data[stat_cols].values
        team_b_lstm = team_b_data[lstm_cols].values
        
        # Combine features: [TeamA_stats, TeamA_LSTM, TeamB_stats, TeamB_LSTM]
        features = np.concatenate([team_a_stats, team_a_lstm, team_b_stats, team_b_lstm])
        
        # Label: 1 if team A won, 0 if team A lost
        label = 1 if result == 'W' else 0
        
        matchups.append({
            'team_a': team_a,
            'team_b': team_b_data['team'],
            'features': features.tolist(),
            'label': label,
            'date': game['date']
        })
    
    print(f"Created {len(matchups)} matchup records")
    
    if len(matchups) == 0:
        print("WARNING: No matchups created! Check opponent name matching.")
        return
    
    # Convert to DataFrame
    matchups_df = pd.DataFrame(matchups)
    
    # Split features into separate columns for database storage
    feature_dim = len(matchups[0]['features'])
    feature_cols_names = [f'feature_{i}' for i in range(feature_dim)]
    
    features_expanded = pd.DataFrame(
        matchups_df['features'].tolist(),
        columns=feature_cols_names
    )
    
    # Combine with metadata
    final_df = pd.concat([
        matchups_df[['team_a', 'team_b', 'label', 'date']],
        features_expanded
    ], axis=1)
    
    # Save to database
    final_df.to_sql('matchup_features', engine, if_exists='replace', index=False)
    
    print(f"✓ Saved {len(final_df)} matchup feature vectors")
    print(f"✓ Feature dimension: {feature_dim}")
    print(f"✓ Win rate in dataset: {final_df['label'].mean():.2%}")
    
    context['ti'].xcom_push(key='matchups_created', value=len(final_df))

class GamePredictor(nn.Module):
    """Neural network to predict game outcomes"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super(GamePredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_neural_net_predictor(**context):
    """Task 5: Train neural network to predict game outcomes"""
    print("Training neural network predictor...")
    
    engine = create_engine(DB_CONN)
    
    # Load matchup features
    df = pd.read_sql('SELECT * FROM matchup_features', engine)
    print(f"Loaded {len(df)} matchup records")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = GamePredictor(input_dim=input_dim, hidden_dims=[256, 128, 64])
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Move data to device
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)
    
    # Training loop
    epochs = 100
    batch_size = 16
    best_accuracy = 0.0
    
    print("\nTraining neural network...")
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        
        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train_t[batch_indices]
            batch_y = y_train_t[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_predictions = (test_outputs > 0.5).float()
                test_accuracy = (test_predictions == y_test_t).float().mean().item()
                
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                
                avg_loss = epoch_loss / (len(X_train_t) // batch_size + 1)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Training set
        train_outputs = model(X_train_t)
        train_predictions = (train_outputs > 0.5).float().cpu().numpy()
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        # Test set
        test_outputs = model(X_test_t)
        test_predictions = (test_outputs > 0.5).float().cpu().numpy()
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print("\n" + "="*50)
        print("NEURAL NETWORK RESULTS")
        print("="*50)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_predictions, target_names=['Loss', 'Win']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_predictions))
    
    # Save model
    model_path = Path('/opt/airflow/data') / 'game_predictor_nn.pt'
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_cols': feature_cols
    }, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save results to database
    results_df = pd.DataFrame([{
        'model_type': 'neural_network',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'best_test_accuracy': best_accuracy,
        'num_features': input_dim,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }])
    results_df.to_sql('model_results', engine, if_exists='replace', index=False)
    print("✓ Results saved to database")
    
    context['ti'].xcom_push(key='nn_test_accuracy', value=test_accuracy)

def train_traditional_ml_models(**context):
    """Task 6: Train traditional ML models for comparison"""
    print("Training traditional ML models...")
    
    engine = create_engine(DB_CONN)
    
    # Load matchup features
    df = pd.read_sql('SELECT * FROM matchup_features', engine)
    print(f"Loaded {len(df)} matchup records")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Same train/test split as neural network (for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Define models to train
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Loss', 'Win']))
        
        # Save results
        results.append({
            'model_type': model_name.lower().replace(' ', '_'),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'num_features': X_train.shape[1],
            'train_size': len(X_train),
            'test_size': len(X_test)
        })
        
        # Save model
        model_path = Path('/opt/airflow/data') / f'model_{model_name.lower().replace(" ", "_")}.pkl'
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Model saved to {model_path}")
    
    # Load neural network results for comparison
    nn_results = pd.read_sql("SELECT * FROM model_results WHERE model_type = 'neural_network'", engine)
    
    # Combine all results
    all_results = pd.DataFrame(results)
    all_results = pd.concat([nn_results, all_results], ignore_index=True)
    
    # Save to database
    all_results.to_sql('model_results', engine, if_exists='replace', index=False)
    
    # Print comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    comparison = all_results[['model_type', 'train_accuracy', 'test_accuracy']].sort_values('test_accuracy', ascending=False)
    print(comparison.to_string(index=False))
    print("="*70)
    
    context['ti'].xcom_push(key='best_model', value=comparison.iloc[0]['model_type'])
    context['ti'].xcom_push(key='best_accuracy', value=comparison.iloc[0]['test_accuracy'])

def generate_predictions(**context):
    """Task 7: Generate daily predictions using ensemble of all models"""
    print("Generating predictions for upcoming Big 12 games...")
    
    engine = create_engine(DB_CONN)
    
    # Load team data
    team_stats = pd.read_sql('SELECT * FROM team_current_stats', engine)
    lstm_features = pd.read_sql('SELECT * FROM lstm_features', engine)
    team_data = team_stats.merge(lstm_features, on='team', how='inner')
    
    # Load all games to find upcoming matchups
    games_df = pd.read_sql('SELECT * FROM game_logs', engine)
    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')
    
    # Find future games
    future_games = games_df[games_df['result'].isna() | (games_df['result'] == '')]
    future_games = future_games.dropna(subset=['date'])
    future_games = future_games.sort_values('date')
    
    print(f"Found {len(future_games)} upcoming games")
    
    # Load ALL models
    import pickle
    models = {}
    model_accuracies = {}
    
    # Load model results to get accuracies
    model_results = pd.read_sql('SELECT * FROM model_results', engine)
    
    # Load traditional ML models
    for model_name in ['random_forest', 'gradient_boosting', 'xgboost', 'logistic_regression']:
        model_path = Path('/opt/airflow/data') / f'model_{model_name}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
            acc_row = model_results[model_results['model_type'] == model_name]
            if len(acc_row) > 0:
                model_accuracies[model_name] = acc_row['test_accuracy'].values[0]
    
    # Load neural network
    nn_path = Path('/opt/airflow/data') / 'game_predictor_nn.pt'
    if nn_path.exists():
        checkpoint = torch.load(nn_path)
        nn_model = GamePredictor(input_dim=checkpoint['input_dim'])
        nn_model.load_state_dict(checkpoint['model_state'])
        nn_model.eval()
        models['neural_network'] = nn_model
        model_accuracies['neural_network'] = checkpoint['test_accuracy']
    
    print(f"\nLoaded {len(models)} models:")
    for name, acc in sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {acc:.4f}")
    
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    print(f"\nBest individual model: {best_model_name} ({model_accuracies[best_model_name]:.4f})")
    
    # Prepare feature columns
    stat_cols = ['avg_points', 'avg_opp_points', 'avg_fg_pct', 'avg_three_pct', 
                 'avg_ft_pct', 'avg_rebounds', 'avg_assists', 'avg_steals', 
                 'avg_blocks', 'avg_turnovers']
    lstm_cols = [f'lstm_f{i}' for i in range(LSTM_FEATURE_DIM)]
    
    predictions = []
    seen_matchups = set()
    
    for idx, game in future_games.head(20).iterrows():
        team_a = game['team']
        team_b = game['opponent']
        game_date = game['date']
        
        # Get team data
        team_a_data = team_data[team_data['team'] == team_a]
        if len(team_a_data) == 0:
            continue
        
        team_b_candidates = team_data[team_data['team'].str.contains(team_b.split()[0], case=False, na=False)]
        if len(team_b_candidates) == 0:
            continue
        
        team_b_data = team_b_candidates.iloc[0]
        team_b_full = team_b_data['team']
        
        # Deduplicate
        matchup_id = tuple(sorted([team_a, team_b_full]))
        if matchup_id in seen_matchups:
            continue
        seen_matchups.add(matchup_id)
        
        # Build features
        team_a_stats = team_a_data[stat_cols].values.flatten()
        team_a_lstm = team_a_data[lstm_cols].values.flatten()
        team_b_stats = team_b_data[stat_cols].values
        team_b_lstm = team_b_data[lstm_cols].values
        
        features = np.concatenate([team_a_stats, team_a_lstm, team_b_stats, team_b_lstm])
        features = features.astype(np.float32)
        
        if np.isnan(features).any():
            continue
        
        # Get predictions from ALL models
        model_predictions = {}
        
        for model_name, model in models.items():
            if model_name == 'neural_network':
                # Neural network
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    prob = model(features_tensor).item()
                model_predictions[model_name] = prob
            else:
                # Traditional ML models
                pred_proba = model.predict_proba(features.reshape(1, -1))[0]
                model_predictions[model_name] = pred_proba[1]
        
        # Weighted ensemble: weight by test accuracy
        total_weight = sum(model_accuracies.values())
        weighted_prob = sum(
            model_predictions[name] * model_accuracies[name] 
            for name in model_predictions
        ) / total_weight
        
        # Simple majority vote
        votes_for_team_a = sum(1 for prob in model_predictions.values() if prob > 0.5)
        
        # Get team stats
        team_a_ppg = float(team_a_data['avg_points'].values[0])
        team_b_ppg = float(team_b_data['avg_points'])
        team_a_record = f"{int(team_a_data['wins'].values[0])}-{int(team_a_data['losses'].values[0])}"
        team_b_record = f"{int(team_b_data['wins'])}-{int(team_b_data['losses'])}"
        
        predicted_winner = team_a if weighted_prob > 0.5 else team_b_full
        confidence = weighted_prob if weighted_prob > 0.5 else (1 - weighted_prob)
        
        predictions.append({
            'date': game_date,
            'team_a': team_a,
            'team_b': team_b_full,
            'team_a_ppg': team_a_ppg,
            'team_b_ppg': team_b_ppg,
            'team_a_record': team_a_record,
            'team_b_record': team_b_record,
            'predicted_winner': predicted_winner,
            'ensemble_confidence': confidence,
            'weighted_prob': weighted_prob,
            'votes_for_a': votes_for_team_a,
            'total_votes': len(model_predictions),
            'model_predictions': str(model_predictions)  # For debugging
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    if len(predictions_df) > 0:
        predictions_df = predictions_df.sort_values('date')
        predictions_df.to_sql('daily_predictions', engine, if_exists='replace', index=False)
        
        # Enhanced report
        report = []
        report.append("="*80)
        report.append(f"BIG 12 BASKETBALL PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
        report.append("="*80)
        report.append(f"Ensemble Model: {len(models)} models voting")
        report.append(f"Best Individual: {best_model_name.replace('_', ' ').title()} ({model_accuracies[best_model_name]:.1%})")
        report.append("="*80)
        report.append("")
        
        for idx, pred in predictions_df.iterrows():
            date_str = pred['date'].strftime('%a, %b %d')
            report.append(f"{date_str}: {pred['team_a']} vs {pred['team_b']}")
            report.append(f"  {pred['team_a']} ({pred['team_a_record']}, {pred['team_a_ppg']:.1f} PPG) vs {pred['team_b']} ({pred['team_b_record']}, {pred['team_b_ppg']:.1f} PPG)")
            report.append(f"  🏆 ENSEMBLE PICK: {pred['predicted_winner']}")
            report.append(f"  📊 Confidence: {pred['ensemble_confidence']:.1%}")
            report.append(f"  🗳️  Model Vote: {pred['votes_for_a']}/{pred['total_votes']} models pick {pred['team_a']}")
            report.append(f"  🎲 Win Probability: {pred['team_a']} {pred['weighted_prob']:.1%} | {pred['team_b']} {1-pred['weighted_prob']:.1%}")
            report.append("")
        
        report.append("="*80)
        report.append(f"Total Predictions: {len(predictions_df)}")
        report.append(f"Prediction Method: Accuracy-weighted ensemble")
        report.append("="*80)
        report_text = "\n".join(report)
        
        report_path = Path('/opt/airflow/data/raw') / f'predictions_{datetime.now().strftime("%Y%m%d")}.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to {report_path}")
        
    else:
        print("No upcoming matchups found")
    
    context['ti'].xcom_push(key='predictions_generated', value=len(predictions_df))

# Task 1: Scrape team stats
task_scrape = PythonOperator(
    task_id='scrape_team_stats',
    python_callable=scrape_team_stats,
    dag=dag,
)

# Task 2: Calculate rolling averages
task_averages = PythonOperator(
    task_id='calculate_team_averages',
    python_callable=calculate_team_averages,
    dag=dag,
)

# Task 3: LSTM encoder and plots
task_lstm = PythonOperator(
    task_id='train_lstm_encoder',
    python_callable=train_lstm_encoder,
    dag=dag,
)

# Task 4: LSTM analysis
task_lstm_analysis = PythonOperator(
    task_id='analyze_lstm_features',
    python_callable=analyze_lstm_features,
    dag=dag,
)

# Task 5: Matchups
task_matchups = PythonOperator(
    task_id='assemble_matchup_features',
    python_callable=assemble_matchup_features,
    dag=dag,
)

# Task 6: train nn
task_train_nn = PythonOperator(
    task_id='train_neural_net_predictor',
    python_callable=train_neural_net_predictor,
    dag=dag,
)

# Task 7: train ml
task_train_ml = PythonOperator(
    task_id='train_traditional_ml_models',
    python_callable=train_traditional_ml_models,
    dag=dag,
)

# Task 8: predictions
task_predictions = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag,
)

# Set task dependencies
task_scrape >> task_averages >> task_lstm >> task_lstm_analysis >> task_matchups >> task_train_nn >> task_train_ml >> task_predictions