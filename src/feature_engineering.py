# src/feature_engineering.py
import pandas as pd
import numpy as np
from typing import List

# --- columns we expect in your dataset (all lowercase) ---
BASE_COLS = [
    "player_id","player_name","matches_played","kills","deaths","assists",
    "damage","headshots","wins","top10s","revives","distance",
    "weapons_used","survival_time","rank"
]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure all expected columns exist; if missing, create filled with 0."""
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    for c in BASE_COLS:
        if c not in out.columns:
            out[c] = 0
    # numeric
    for c in [c for c in BASE_COLS if c not in ("player_id","player_name")]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    # avoid zero matches for rate features
    out["matches_played"] = out["matches_played"].replace(0, 1)
    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-match, composite and target features using your 'distance' column."""
    out = _ensure_cols(df)

    eps = 1e-6
    # per-match rates
    out["kills_per_match"]     = out["kills"]     / (out["matches_played"] + eps)
    out["assists_per_match"]   = out["assists"]   / (out["matches_played"] + eps)
    out["damage_per_match"]    = out["damage"]    / (out["matches_played"] + eps)
    out["revives_per_match"]   = out["revives"]   / (out["matches_played"] + eps)
    out["survival_per_match"]  = out["survival_time"] / (out["matches_played"] + eps)
    out["movement_per_match"]  = out["distance"]  / (out["matches_played"] + eps)
    out["headshot_rate"]       = out["headshots"] / (out["kills"] + eps)

    # composite scores
    out["aggression_score"] = (
        2.5*out["kills_per_match"] + out["damage_per_match"]/150.0 + 1.5*out["headshot_rate"]
    )
    out["support_score"] = 1.5*out["assists_per_match"] + 2.0*out["revives_per_match"]
    out["survival_score"] = out["survival_per_match"]/5.0 + out["movement_per_match"]/2000.0

    out["wpi"] = 0.5*out["aggression_score"] + 0.3*out["survival_score"] + 0.2*out["support_score"]

    # synthetic auction_value (replace with real labels if you have them)
    out["auction_value"] = (50*out["wpi"] + 2*out["kills_per_match"]
                            + 1*out["assists_per_match"] + out["damage_per_match"]/100.0) * 10
    return out

FEATURES_FOR_MODEL: List[str] = [
    # raw
    "matches_played","kills","assists","damage","headshots","revives",
    "survival_time","distance",
    # engineered
    "kills_per_match","assists_per_match","damage_per_match","headshot_rate",
    "revives_per_match","survival_per_match","movement_per_match",
    "aggression_score","support_score","survival_score","wpi"
]
