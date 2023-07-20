import pandas as pd
from assets.train_bertopic import train_model


if __name__ == '__main__':
    # Load data from Costumer Service Chat Data 30k Rows
    data = pd.read_excel("data/CustomerServiceChatData30kRows.xlsx")
    print("Number of texts: ", len(data))
    print(data)
    docs = data["Text"]
    original_len = len(docs)

    # drop empty text columns
    docs = docs.dropna().reset_index(drop=True)
    print(f"original len {original_len}, len after dropping nas {len(docs)}, dropped {original_len-len(docs)} columns")

    # drop columns that contained just numbers (not strings)
    print(sum([not isinstance(text, str) for text in docs]))
    docs = docs[[isinstance(text, str) for text in docs]].reset_index(drop=True)
    print(len(docs))

    # Example: load data from csv
    # data = pd.read_csv("<file_name>.csv")
    # docs = data["<column_name>"]

    # train model
    model = train_model(docs, min_df = 3, save_model=True, model_name="kaggle_data", n_gram=(1,3), get_metrics=False)


