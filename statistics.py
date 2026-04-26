import pandas as pd

df = pd.read_json("transactions.json")
games_df = df.drop(columns=["transaction_id"])
game_popularity = games_df.sum().sort_values(ascending=False)

total_users = df.shape[0]
total_games = games_df.shape[1]

print("Общая статистика:")
print("Всего пользователей:", total_users)
print("Всего игр:", total_games)

print("\nТОП-10 игр:")
print(game_popularity.head(10))

df["games_count"] = games_df.sum(axis=1)

print("\nСтатистика по пользователям:")
print("Среднее число игр:", df["games_count"].mean())
print("Максимум игр:", df["games_count"].max())
print("Минимум игр:", df["games_count"].min())

rare_games = game_popularity[game_popularity == game_popularity.min()]

print("\nРедкие игры (минимальная популярность):")
print(rare_games.head(10))
