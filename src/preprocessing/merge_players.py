import pandas as pd
import os


RAW_PATH = "data/raw/"
SAVE_PATH = "data/processed/"


def load_csv(name):
    return pd.read_csv(
        os.path.join(RAW_PATH,name)
    )


# =============================
# LOAD DATA
# =============================

stats = load_csv("player_stats.csv")

shooting = load_csv("player_shooting.csv")

passing = load_csv("player_passing.csv")

possession = load_csv("player_possession.csv")

defense = load_csv("player_defense.csv")

misc = load_csv("player_misc.csv")


# =============================
# SELECT FEATURES
# =============================


stats = stats[
[
"player",
"team",
"position",
"age",
"club",
"games",
"games_starts",
"minutes",
"goals",
"assists",
"xg",
"xg_assist"
]
]


shooting = shooting[
[
"player",
"team",
"shots",
"shots_on_target",
"shots_on_target_pct",
"goals_per_shot",
"average_shot_distance"
]
]


passing = passing[
[
"player",
"team",
"passes_completed",
"passes",
"passes_pct",
"assisted_shots",
"passes_into_final_third",
"passes_into_penalty_area",
"progressive_passes"
]
]


possession = possession[
[
"player",
"team",
"touches",
"dribbles",
"dribbles_completed",
"dribbles_completed_pct",
"progressive_passes_received",
"miscontrols",
"dispossessed"
]
]


defense = defense[
[
"player",
"team",
"tackles",
"tackles_won",
"blocks",
"interceptions",
"clearances"
]
]


misc = misc[
[
"player",
"team",
"fouls",
"fouled",
"ball_recoveries",
"aerials_won",
"aerials_won_pct"
]
]



# =============================
# MERGING
# =============================

dfs=[
    shooting,
    passing,
    possession,
    defense,
    misc
]


players = stats.copy()


for df in dfs:

    players = players.merge(
        df,
        on=["player","team"],
        how="left"
    )


# =============================
# CLEANING
# =============================

# =============================
# HANDLE MISSING VALUES
# =============================


# numerical columns
numeric_cols = players.select_dtypes(
    include=["int64","float64"]
).columns


players[numeric_cols] = players[numeric_cols].fillna(0)



# categorical columns
categorical_cols = players.select_dtypes(
    include=["object","string"]
).columns


players[categorical_cols] = players[categorical_cols].fillna("Unknown")

# =============================
# =============================
# CLEANING
# =============================


# Handle numerical missing values

numeric_cols = players.select_dtypes(
    include=["int64","float64"]
).columns


players[numeric_cols] = (
    players[numeric_cols]
    .fillna(0)
)



# Handle categorical missing values

categorical_cols = players.select_dtypes(
    include=["object","string"]
).columns


players[categorical_cols] = (
    players[categorical_cols]
    .fillna("Unknown")
)



# Convert age format
# Example: "35-123" -> 35


players["age_years"] = (
    players["age"]
    .astype(str)
    .str.split("-")
    .str[0]
    .astype(int)
)


players.drop(
    columns=["age"],
    inplace=True
)


# =============================
# FEATURE ENGINEERING
# =============================


players["attacking_score"] = (

    players["goals"]*4
    +
    players["assists"]*3
    +
    players["shots_on_target"]*0.5
    +
    players["xg"]

)



players["playmaking_score"] = (

    players["assisted_shots"]
    +
    players["progressive_passes"]
    +
    players["passes_into_final_third"]

)



players["possession_score"] = (

    players["dribbles_completed"]
    +
    (players["touches"]*0.01)
    -
    players["miscontrols"]
    -
    players["dispossessed"]

)



players["defensive_score"] = (

    players["tackles_won"]
    +
    players["interceptions"]
    +
    players["blocks"]
    +
    players["clearances"]

)



players["impact_score"] = (

    players["attacking_score"]
    +
    players["playmaking_score"]
    +
    players["possession_score"]
    +
    players["defensive_score"]

)



# =============================
# SAVE
# =============================


os.makedirs(
    SAVE_PATH,
    exist_ok=True
)


players.to_csv(
    os.path.join(SAVE_PATH,"players_final.csv"),
    index=False
)



print("Player dataset created successfully")
print(players.shape)
print(players.head())