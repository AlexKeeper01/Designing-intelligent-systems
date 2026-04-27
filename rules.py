import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.4

df = pd.read_json("transactions.json")
X = df.drop(columns=["transaction_id"])
itemsets = apriori(
    X,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

rules = association_rules(
    itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

rules = rules[["antecedents", "consequents", "support", "confidence", "conviction", "lift", "leverage"]]
rules = rules.sort_values(by="lift", ascending=False)

rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

print(rules[["antecedents", "consequents", "support", "confidence", "conviction", "lift", "leverage"]].to_string(index=False)
)
