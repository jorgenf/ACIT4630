import csv
import pandas as pd

# PARAMS
ECO_THRESHOLD = 100  # min occurence of a opening
features = [  # columns to be copies to new dataset
    "winner",  # 0 black, 1 white
    "turns",
    "white_rating",
    "black_rating",
    "relative_rating",
    "opening_ply"
]

# misc
dataset = []

# read csv-file into pandas dataframe
df = pd.read_csv("games.csv")

# remove rare ECO-codes
df = df[df.groupby("opening_eco").opening_eco.transform("count") > ECO_THRESHOLD].copy()
features += sorted(df.opening_eco.unique())

# delete draws
df = df[df.winner != "draw"]

# delete unrated games
df = df[df.rated != True]

# iterate over df and copy values to new dataset
for row in df.itertuples():
    new_row = dict.fromkeys(features, 0)

    # code winner-label
    if row.winner == "white":
        new_row["winner"] = 1
    elif row.winner == "black":
        new_row["winner"] = 0
    else:
        print("error?", row.winner)
        continue

    new_row["turns"] = row.turns
    new_row["white_rating"] = row.white_rating
    new_row["black_rating"] = row.black_rating
    new_row["relative_rating"] = int(row.white_rating) - int(row.black_rating)
    new_row["opening_ply"] = row.opening_ply

    eco_code = str(row.opening_eco)
    new_row[eco_code] = 1

    dataset.append(new_row)

print(len(dataset))

# write new dataset
with open("new_dataset.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=features)
    writer.writeheader()
    writer.writerows(dataset)
