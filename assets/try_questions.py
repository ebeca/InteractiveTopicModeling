import pandas as pd

df = pd.read_csv("questions.csv")

list_questions = df["option1"] + df["option2"] + df["option2"] + df["option3"]

sorted_list = sorted(list_questions, key=len)

for i in range(5):
    print(sorted_list[i])