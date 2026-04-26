import kagglehub
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


path = kagglehub.dataset_download("tamber/steam-video-games")
df = pd.read_csv(path + "/steam-200k.csv", header=None)
df.columns = ["user_id", "game", "action", "hours", "unknown"]

df = df[df["action"] == "play"]
df = df[df["hours"] > 5]

game_counts = df["game"].value_counts()
valid_games = game_counts[game_counts > 20].index
df = df[df["game"].isin(valid_games)]

transactions = df.groupby("user_id")["game"].apply(list)
transactions = transactions[transactions.apply(len) >= 2]
transactions_list = transactions.tolist()
te = TransactionEncoder()
te_array = te.fit(transactions_list).transform(transactions_list)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

df_encoded = df_encoded.reset_index(drop=True)
df_encoded.insert(0, "transaction_id", df_encoded.index)
df_encoded.to_json("transactions.json", orient="records", indent=2)

print("Количество транзакций:", len(df_encoded))
print("Количество игр:", len(df_encoded.columns) - 1)
