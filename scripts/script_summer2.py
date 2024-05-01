import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from IPython.display import display, HTML
import random
import numpy as np
import datasets
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq
from evaluate import load
metric = load("rouge")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

class Preprocessor2:
    def __init__(
        self,
        remove_url=True,
        remove_punct=True,
        remove_stopwords=True,
        tokenize_words=True,
        lemmatize_words=True,
        background_color=True,
        show_df=True,
        show_random_elements=True,
        preprocess_function=True,
        compute_metrics=True,
        split_dataset_to_dataframes=True
    ) -> None:
        self.methods = []
        if remove_url:
            self.methods.append(self._remove_URL)
        if remove_punct:
            self.methods.append(self._remove_punct)
        if remove_stopwords:
            nltk.download("stopwords")
            self.methods.append(self._remove_stopwords)
        if tokenize_words:
            nltk.download("punkt")
            self.methods.append(self._tokenize_words)
        if lemmatize_words:
            nltk.download("wordnet")
            self.methods.append(self._lemmatize_words)
        if background_color:
            self.methods.append(self.background_color)
        if show_df:
            self.methods.append(self.show_df)
        if show_random_elements:
             self.methods.append(self.show_random_elements)
        if preprocess_function:
             self.methods.append(self.preprocess_function)
        if compute_metrics:
            self.methods.append(self.compute_metrics)
        if split_dataset_to_dataframes:
            self.methods.append(self.split_dataset_to_dataframes)

    def apply(self, data: pd.Series | str | list) -> str | list[str]:
        return self(data)

    def __call__(self, data: str | list) -> str | list[str]:
        """Apply cleaning methods on the data and return the cleaned data"""
        result = data
        for method in self.methods:
            if isinstance(data, pd.Series):
                result = result.map(method)
            else:
                result = method(result)
        return result

    def _remove_URL(self, text: str) -> str:
        url = re.compile(r"https?://\S+|www\.\S+")
        return url.sub(r"", text)

    def _remove_punct(self, text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def _remove_stopwords(self, text: str) -> str:
        stop = set(stopwords.words("english"))

        filtered_words = [
            word.lower() for word in text.split() if word.lower() not in stop
        ]
        return " ".join(filtered_words)

    def _tokenize_words(self, text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text)

    def _lemmatize_words(self, text: list[str]) -> list[str]:
        lemmatizer = WordNetLemmatizer()
        result_sentence = []
        for token in text:
            result_sentence.append(lemmatizer.lemmatize(token))
        return result_sentence

    def background_color(self, value):
        if isinstance(value, str):
            return 'background-color: #a6c0ed'
        return ''

    def show_df(self, df_train):
        print('shape'.center(30,'_'))
        display(df_train)

        print('head'.center(30,'_'))
        display(df_train.head().style.background_gradient(cmap='Blues').applymap(self.background_color))

        print('tail'.center(30,'_'))
        display(df_train.tail().style.background_gradient(cmap='Blues').applymap(self.background_color))

        print('info'.center(30,'_')+'\n')
        display(df_train.info())

        print('describe_continuous'.center(30,'_'))
        display(df_train.describe().T.style.background_gradient(cmap = 'Blues'))

        print('describe_categorical'.center(30,'_'))
        display(df_train.describe(include='object').T.applymap(self.background_color))

        print('null_values_percent'.center(30,'_'))
        display((df_train.isna().sum() / len(df_train) * 100).sort_values(ascending=False))

    def show_random_elements(self, dataset, num_examples=5):
        assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)

        df = pd.DataFrame(dataset[picks])
        for column, typ in dataset.features.items():
            if isinstance(typ, datasets.ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
        display(HTML(df.to_html()))

    max_input_length = 1024
    max_target_length = 128

    prefix = "summarize: "

    def preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

   

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    

    def split_dataset_to_dataframes(self, dataset):
        train_df = pd.DataFrame()
        validation_df = pd.DataFrame()
        test_df = pd.DataFrame()

        for split_name, split_data in dataset.items():
            if 'train' in split_name.lower():
                train_df = pd.concat([train_df, split_data.to_pandas()])
            elif 'validation' in split_name.lower():
                validation_df = pd.concat([validation_df, split_data.to_pandas()])
            elif 'test' in split_name.lower():
                test_df = pd.concat([test_df, split_data.to_pandas()])

        print("Train DataFrame:")
        print(train_df)
        print("\nValidation DataFrame:")
        print(validation_df)
        print("\nTest DataFrame:")
        print(test_df)

        return train_df, validation_df, test_df


