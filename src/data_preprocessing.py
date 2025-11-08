# src/data_preprocessing.py (only key parts changed)
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/pubg_stats.csv")
PROCESSED = Path("data/processed/players_processed.csv")

RENAME = {
    # map any variants to our canonical names
    "Player_Name":"player_name", "Matches_Played":"matches_played",
    "Kills":"kills","Deaths":"deaths","Assists":"assists","Damage_Dealt":"damage",
    "Damage":"damage","Headshots":"headshots","Wins":"wins","Top_10s":"top10s",
    "Top10s":"top10s","Revives":"revives","Distance_Traveled":"distance",
    "Distance":"distance","Weapons_Used":"weapons_used","Time_Survived":"survival_time",
    "Survival_Time":"survival_time","Rank":"rank"
}

KEEP = ["player_id","player_name","matches_played","kills","deaths","assists","damage",
        "headshots","wins","top10s","revives","distance","weapons_used","survival_time","rank"]

def load_and_basic_clean(path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME)
    df.columns = df.columns.str.lower()

    # ensure essential ids
    if "player_id" not in df.columns:
        # if no explicit id, create from name
        if "player_name" in df.columns:
            df["player_id"] = df["player_name"].str.lower().str.replace(r"\s+","_", regex=True)
        else:
            raise AssertionError("player_id column missing and no player_name to derive it.")

    # keep only relevant
    df = df[[c for c in KEEP if c in df.columns]].copy()

    # numeric coercion
    for c in df.columns:
        if c not in ("player_id","player_name"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # clip big outliers a bit
    for c in ["kills","assists","damage","headshots","revives","survival_time","distance"]:
        if c in df:
            hi = df[c].quantile(0.995)
            df[c] = df[c].clip(lower=0, upper=hi)

    return df
