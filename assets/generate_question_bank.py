import pandas as pd
from assets.train_bertopic import BERTopicAdjusted
import random as rd

QUESTIONS_PER_TOPIC = 2
NUMBER_OF_OPTIONS = 4  # MC options per question

rd.seed(21984)

model = BERTopicAdjusted.load("../models/kaggle_data")
topic_indexes = model.get_topic_info()["Topic"]
list_indexes = topic_indexes.to_list()

# list for the options
option1 = []
option2 = []
option3 = []
option4 = []
# list for correct answer (intruder)
correct_answers = []

for t in topic_indexes:
    # copy of the indexes without current index
    copy_indexes = list_indexes[:]
    copy_indexes.remove(t)
    documents = model.get_representative_docs(topic=t)
    documents = list(dict.fromkeys(documents))  # remove duplicate documents
    documents = [d for d in documents if 10 <= len(d) < 200] # filter documents too short or too long

    if len(documents) > NUMBER_OF_OPTIONS - 1:  # there are enough documents
        for _ in range(QUESTIONS_PER_TOPIC):

            options = rd.sample(documents, NUMBER_OF_OPTIONS - 1)

            # choosing a random topic to pick the intruder from
            topic_intruder = rd.choice(copy_indexes)

            # choosing a random document from intruder topic to be the intruder
            topic_intruder_docs = model.get_representative_docs(topic=topic_intruder)
            intruder = rd.choice(topic_intruder_docs)

            # adding all options to the lists
            options.append(intruder)
            rd.shuffle(options)
            opt1, opt2, opt3, opt4 = options
            option1.append(opt1)
            option2.append(opt2)
            option3.append(opt3)
            option4.append(opt4)

            correct_answers.append(intruder)

print(len(option1), len(option2), len(option3), len(option4), len(correct_answers))

questions = ["Select the intruder"] * len(option1)
required = [True] * len(option1)
question_type = ["multiple choice"] * len(option1)

# Create dataframe for the questions
df = pd.DataFrame({'questions': questions,
                   'question type': question_type,
                   'required': required,
                   'option1': option1,
                   'option2': option2,
                   'option3': option3,
                   'option4': option4,
                   'correct answer': correct_answers
                   })

df.to_csv("questions.csv")
