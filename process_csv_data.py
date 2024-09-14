# import nltk
# nltk.download('wordnet')
# import necessary libraries
import pandas as pd
import json

# define preprocessor
from nltk.stem import WordNetLemmatizer
import gensim.parsing.preprocessing as gpp
import gensim.utils as gu
import unicodedata

def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')  # this line removes special characters
    preprocs = [
        gpp.strip_tags, 
        gpp.strip_punctuation,
        gpp.strip_multiple_whitespaces,
        gpp.strip_numeric,
        gpp.remove_stopwords, 
        gpp.strip_short, 
    ]
    text = gu.to_unicode(text.lower().strip())
    for preproc in preprocs:
        text = preproc(text)
    return text

def lemmatize(text):
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(text)  

preprocess = lambda text: lemmatize(preprocess_text(str(text)))


# define data array
data = [
    {
        "source": "data/HighValue_Positive-Reviews.csv",
        "completion": " high_value_positive"
    },
    {
        "source": "data/HighValue_MidLevel-Reviews.csv",
        "completion": " high_value_middle"
    },
    {
        "source": "data/HighValue_Negative-Reviews.csv",
        "completion": " high_value_negative"
    },
    {
        "source": "data/LowValue_Reviews.csv",
        "completion": " low_value"
    }
]

# define chunk size
chunk_size = 5000 # adjust according to your needs and available memory

# opening the output file
with open("prompts.jsonl", "a") as outfile:  # 'a' for append mode
    # loop over your data list
    for file_data in data:
        # get file source and completion from the current data dict
        source, completion = file_data["source"], file_data["completion"]

        # create chunks iterator
        chunks = pd.read_csv(source, header=None, chunksize=chunk_size)

        # loop over each chunk
        for chunk in chunks:
            # convert the chunk to a 1-D list
            reviews = chunk.values.ravel()

            for review_text in reviews:
                # check if the review_text is not nan (empty)
                if pd.notna(review_text):
                    # apply preprocessing to the review_text
                    preprocessed_review_text = preprocess(review_text)

                    prompt = preprocessed_review_text + " ->"
                    prompt_completion_pair = {
                        "messages": [
                            {"role": "system", "content": "You are an Amazon Review Analysis chatbot. Your purpose is to quickly label raw reviews as either High Value or Low Value for further processing.Focus on identifying reviews that provide valuable insights for product improvement, regardless of the reviewer's overall sentiment. Pay close attention to reviews that describe the reviewer's experience before and after using the product. Look for indications of a clear change in their emotional state, level of satisfaction, or ability to solve a problem. These reviews can provide valuable insights into the product's impact on users' lives and their overall emotional journey.Consider the following attributes of high-value reviews:-Specificity: Does the review provide specific details about the product's features, benefits, or drawbacks?-Context: Does the review offer context about the reviewer's situation and needs?-Actionable Insights: Does the review offer actionable insights that can be used to improve the product or customer experience?-Emotional Journey: Does the review indication a clear change in their emotional state, level of satisfaction, or ability to solve a problem resulting from using the product?Thoroughly analyse each review and Label reviews as follows:High Value_Positive: If the review is positive and provides valuable insights.High Value_Neutral: If the review is neutral but provides valuable insights.High Value_Negative: If the review is negative but provides valuable insights.Low Value: If the review lacks valuable insights for product improvement."},
                            {"role": "user", "content": preprocessed_review_text},
                            {"role": "assistant", "content": completion}
                        ]
                    }
                    # write to the jsonl file
                    json.dump(prompt_completion_pair, outfile)
                    outfile.write('\n')

                    # print the prompt and completion
        print("Complete ", source)
