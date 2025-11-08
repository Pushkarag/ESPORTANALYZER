from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import pandas as pd
import sys

# Setup project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import features
from src.feature_engineering import add_features

# Try loading ML model
try:
    from src.predict import ModelService
    svc = ModelService()
except:
    svc = None


app = Flask(__name__)
app.secret_key = "mysecret"

DATA = ROOT / "data" / "processed" / "players_processed.csv"


# -------------------------------------------------------------
# ✅ Load players helper
# -------------------------------------------------------------
def load_players():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# -------------------------------------------------------------
# ✅ Strategy generator (longer + clean)
# -------------------------------------------------------------
def generate_strategy(p):
    tips = []

    if p["aggression_score"] > 2.0 and p["survival_score"] < 1.0:
        tips.append("You are too aggressive and die early. Take fights with cover, avoid ego peeks, and reset fights instead of pushing blindly.")

    if p["aggression_score"] < 1.0:
        tips.append("Increase your aggression by taking 1–2 controlled fights per match, practicing recoil, and using shoulder-peeks.")

    if p["support_score"] < 0.6:
        tips.append("Work on support: help with revives, call rotations, use more utility, and watch teammate health during fights.")

    if p["survival_score"] < 0.9:
        tips.append("Improve survival: avoid hot-drops, rotate early, play ridges, avoid open-field pushes, and keep 3 smokes at all times.")

    if p["headshot_rate"] < 0.20:
        tips.append("Low headshot rate. Do flick drills for 10 minutes daily. Focus on burst tapping with 3x or 4x.")

    if p["damage_per_match"] < 700:
        tips.append("Increase damage through long-range taps, spraying vehicles, and taking more mid-range control positions.")

    if p["movement_per_match"] < 2000:
        tips.append("Move more between compounds, scout zones actively, and choose areas with natural third-party opportunities.")

    tips.append("Review your gameplay replays weekly to fix peeking mistakes, slow reactions, and poor positioning in late zone.")

    return tips


# -------------------------------------------------------------
# ✅ Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/player")
def player_redirect():
    q = request.args.get("q", "").strip()
    if q == "":
        flash("Enter a player name")
        return redirect(url_for("index"))
    return redirect(url_for("player", player_name=q))


@app.route("/player/<player_name>")
def player(player_name):
    df = load_players()
    feat = add_features(df)

    row_df = feat[feat["player_name"].str.lower() == player_name.lower()]
    if row_df.empty:
        flash("Player not found.")
        return redirect(url_for("index"))

    row = row_df.iloc[0].to_dict()

    # Fix missing auction value
    if "auction_value" not in row or pd.isna(row.get("auction_value")):
        row["auction_value"] = None

    # Prediction payload (safe fallback)
    payload = {
        "player_id": str(row["player_name"]),
        "matches_played": float(row["matches_played"]),
        "kills": float(row["kills"]),
        "assists": float(row["assists"]),
        "damage": float(row["damage"]),
        "headshots": float(row["headshots"]),
        "revives": float(row["revives"]),
        "survival_time": float(row["survival_time"]),
        "walk_distance": float(row["distance"]),
        "ride_distance": 0.0
    }

    pred = None
    if svc:
        try:
            pred = round(svc.predict(payload), 2)
            row["auction_value"] = pred
        except:
            pred = None

    # Strategy
    tips = generate_strategy(row)

    # Bar chart
    bar_labels = ["Kills/Match", "Assists/Match", "Damage/Match", "Revives/Match"]
    bar_values = [
        row["kills_per_match"],
        row["assists_per_match"],
        row["damage_per_match"],
        row["revives_per_match"]
    ]

    # Radar chart
    radar_labels = ["Aggression", "Support", "Survival"]
    radar_values = [
        row["aggression_score"],
        row["support_score"],
        row["survival_score"]
    ]

    return render_template(
        "player.html",
        row=row,
        pred=pred,
        bar_labels=bar_labels,
        bar_values=bar_values,
        radar_labels=radar_labels,
        radar_values=radar_values,
        tips=tips
    )


# -------------------------------------------------------------
# ✅ Leaderboard
# -------------------------------------------------------------
@app.route("/leaderboard")
def leaderboard():
    df = load_players()
    feat = add_features(df)

    ranked = feat.sort_values("wpi", ascending=False)

    players = []
    for _, r in ranked.iterrows():
        players.append({
            "name": r["player_name"],
            "wpi": round(r["wpi"], 2),
            "kills": int(r["kills"]),
            "damage": int(r["damage"]),
            "matches": int(r["matches_played"])
        })

    return render_template("leaderboard.html", players=players)


# -------------------------------------------------------------
# ✅ 1v1 Compare
# -------------------------------------------------------------
@app.route("/compare", methods=["GET"])
def compare():
    p1 = request.args.get("p1", "").strip()
    p2 = request.args.get("p2", "").strip()

    df = load_players()
    feat = add_features(df)

    # First visit: show empty page
    if p1 == "" or p2 == "":
        return render_template(
            "compare.html",
            r1=None, r2=None,
            p1=p1, p2=p2,
            stats=[]
        )

    # Get both players
    r1_df = feat[feat["player_name"].str.lower() == p1.lower()]
    r2_df = feat[feat["player_name"].str.lower() == p2.lower()]

    if r1_df.empty or r2_df.empty:
        flash("One or both players not found. Check names.")
        return redirect(url_for("compare"))

    # Convert to dict
    r1 = r1_df.iloc[0].to_dict()
    r2 = r2_df.iloc[0].to_dict()

    # ✅ EXACT STATS YOU HAVE — FIXED
    stats = [
        "kills_per_match",
        "assists_per_match",
        "damage_per_match",
        "survival_per_match",
        "movement_per_match",
        "aggression_score",
        "support_score",
        "survival_score",
        "wpi"
    ]

    # ✅ Guarantee all keys exist
    for s in stats:
        r1.setdefault(s, 0)
        r2.setdefault(s, 0)

    return render_template(
        "compare.html",
        r1=r1, r2=r2,
        p1=p1, p2=p2,
        stats=stats
    )
if __name__ == "__main__":
    app.run(debug=True)
