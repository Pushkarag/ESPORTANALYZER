# E-Sports ML: PUBG Player Auction Value Prediction

## Steps
1) Put your CSV at `data/raw/pubg_stats.csv` (columns: player_id, matches_played, kills, assists, damage, headshots, revives, survival_time, walk_distance, ride_distance).
2) Preprocess:
```bash
python -m src.data_preprocessing
