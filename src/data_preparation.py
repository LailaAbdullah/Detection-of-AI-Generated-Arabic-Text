import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import fasttext
import numpy as np
from collections import Counter
import re
from nltk.stem.isri import ISRIStemmer
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack



#phase1
dsAll = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

dfs = []
for split_name in dsAll.keys():
    df = pd.DataFrame(dsAll[split_name])
    df["split"] = split_name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

print(df_all.info())

human_df = df_all[["original_abstract"]].dropna().rename(columns={"original_abstract": "text"})
human_df["label"] = "human"

ai_columns = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract"
]

ai_dfs = []
for col in ai_columns:
    if col in df_all.columns:
        temp = df_all[[col]].dropna().rename(columns={col: "text"})
        temp["label"] = "ai"
        ai_dfs.append(temp)

ai_df = pd.concat(ai_dfs, ignore_index=True)

final_df = pd.concat([human_df, ai_df], ignore_index=True)

print("Preview of the DataFrame:\n")
print(final_df.info())

print("Missing values per column:")
print(df_all.isna().sum())

print("Number of duplicate rows:", df_all.duplicated().sum())

lengths = df_all["original_abstract"].dropna().astype(str).str.len()
print("Very short (<10):", (lengths < 10).sum())
print("Very long (> 3×average):", (lengths > lengths.mean() * 3).sum())


print(final_df.head())
print("\nTotal number of texts:", final_df.shape[0])
print(final_df["label"].value_counts())

#final_df.to_csv("/Users/lailaalmohaymid/PycharmProjects/data_mining/data/raw.csv", index=False)
df = pd.read_csv("/Users/lailaalmohaymid/PycharmProjects/data_mining/data/raw.csv")
#phase 3
# Word length-frequency distribution
def extract_arabic_words(text):
    text = str(text)

    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return re.findall(r'[\u0600-\u06FF]+', text)
df["words"] = df["text"].apply(extract_arabic_words)

def get_arabic_word_lengths(text):
    words = extract_arabic_words(text)
    return [len(w) for w in words]

def get_word_length_frequency(text):
    lengths = get_arabic_word_lengths(text)
    if not lengths:
        return {}
    return dict(Counter(lengths))

df["word_length_frequency"] = df["text"].apply(get_word_length_frequency)
#Average number of S/P
def split_paragraphs(text):
    text = str(text).strip()
    paragraphs = [p.strip() for p in text.split(r'\s*\n\s*\n\s*|\s*\r\n\s*\r\n\s*') if p.strip()]
    return paragraphs if paragraphs else [text] if text else []

def split_sentences(text):
    text = str(text).strip()
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences

df["paragraphs"] = df["text"].apply(split_paragraphs)
df["sentences"] = df["text"].apply(split_sentences)

df["Total number of sentences"] = df["sentences"].apply(len)
df["Total number of paragraphs "] = df["paragraphs"].apply(len)


def compute_avg_S_per_P(row):
    S = row["Total number of sentences"]
    P = row["Total number of paragraphs "]
    return S / P if P > 0 else 0

df["Average number of S/P"] = df.apply(compute_avg_S_per_P, axis=1)


embedding_model = fasttext.load_model(
    "/Users/lailaalmohaymid/PycharmProjects/data_mining/models/cc.ar.300.bin"
)


def build_corpus_vocab(df: pd.DataFrame, words_col: str = "words"):

    all_words = []

    for row_words in df[words_col]:
        if isinstance(row_words, list):
            all_words.extend(row_words)

    counter = Counter(all_words)
    print(f"Total unique words in corpus: {len(counter)}")
    return counter

corpus_counter = build_corpus_vocab(df, words_col="words")
corpus_vocab = list(corpus_counter.keys())


arabic_pattern = re.compile(r'^[\u0600-\u06FF]+$')

def is_good_token(w: str) -> bool:

    if not isinstance(w, str):
        return False
    if len(w) < 3:
        return False
    if not arabic_pattern.match(w):
        return False
    return True

filtered_vocab = [w for w in corpus_vocab if is_good_token(w)]
print(f"Filtered vocab size: {len(filtered_vocab)}")


word_norms = [
    (w, np.linalg.norm(embedding_model.get_word_vector(w)))
    for w in filtered_vocab
]

top50_words = [
    w for w, _ in sorted(word_norms, key=lambda x: x[1], reverse=True)[:50]
]

top50_set = set(top50_words)

print("Top 50 words (from filtered corpus, sample):", top50_words[:20])


def count_top_embedding_words(word_list, top50_set=top50_set):

    if not isinstance(word_list, list):
        return 0

    return sum(1 for w in word_list if w in top50_set)


def num_words_in_top50(
    df: pd.DataFrame,
    words_col: str = "words",
    feature_col: str = "num_words_in_top50_embedding",
    progress_step: int = 300,
):

    df = df.copy()

    if feature_col not in df.columns:
        df[feature_col] = 0

    df = df.reset_index(drop=True)
    n = len(df)
    print(f"Total rows: {n}")

    for i in range(n):
        row_words = df.at[i, words_col]
        df.at[i, feature_col] = count_top_embedding_words(row_words, top50_set)

        if (i + 1) % progress_step == 0 or i == n - 1:
            print(f"Processed {i + 1}/{n} rows ")

    print("\n(num_words_in_top50_embedding) completed ")
    return df


df = num_words_in_top50(
    df,
    words_col="words",
    feature_col="num_words_in_top50_embedding",
    progress_step=300
)

#Perplexity score
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

def calculate_perplexity_batch(texts, batch_size=16, max_length=256):
    results = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        cleaned_batch = []
        batch_indices = []

        for idx, t in enumerate(batch_texts):
            if pd.notna(t) and str(t).strip():
                cleaned_batch.append(str(t))
                batch_indices.append(idx)

        if not cleaned_batch:
            results.extend([None] * len(batch_texts))
            continue

        try:
            inputs = tokenizer(
                cleaned_batch,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                batch_perplexities = []
                for j in range(len(cleaned_batch)):
                    single_input_ids = input_ids[j:j+1]
                    single_attention_mask = attention_mask[j:j+1]

                    single_output = model(
                        single_input_ids,
                        attention_mask=single_attention_mask,
                        labels=single_input_ids
                    )

                    loss = single_output.loss
                    perplexity = torch.exp(loss).item()
                    batch_perplexities.append(perplexity)

                batch_results = [None] * len(batch_texts)
                for idx, ppl in zip(batch_indices, batch_perplexities):
                    batch_results[idx] = ppl

                results.extend(batch_results)

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            results.extend([None] * len(batch_texts))

    return results

perplexities = calculate_perplexity_batch(df["text"].tolist(), batch_size=16, max_length=256)
df["perplexity"] = perplexities

#GPT-2 Output Probability
def calculate_average_log_prob_batch(texts, batch_size=16, max_length=256):
    results = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        cleaned_batch = []
        batch_indices = []

        for idx, t in enumerate(batch_texts):
            if pd.notna(t) and str(t).strip():
                cleaned_batch.append(str(t))
                batch_indices.append(idx)

        if not cleaned_batch:
            results.extend([None] * len(batch_texts))
            continue

        try:
            inputs = tokenizer(
                cleaned_batch,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                batch_probabilities = []
                for j in range(len(cleaned_batch)):
                    single_input_ids = input_ids[j:j+1]
                    single_attention_mask = attention_mask[j:j+1]

                    if single_input_ids.shape[1] <= 1:
                        batch_probabilities.append(None)
                        continue

                    single_output = model(
                        single_input_ids,
                        attention_mask=single_attention_mask,
                        labels=single_input_ids
                    )

                    average_nll = single_output.loss.item()
                    average_log_prob = -average_nll
                    average_probability = torch.exp(torch.tensor(average_log_prob)).item()
                    batch_probabilities.append(average_probability)

                batch_results = [None] * len(batch_texts)
                for idx, prob in zip(batch_indices, batch_probabilities):
                    batch_results[idx] = prob

                results.extend(batch_results)

        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            results.extend([None] * len(batch_texts))

    return results

probabilities = calculate_average_log_prob_batch(df["text"].tolist(), batch_size=16, max_length=256)
df["gpt2_output_probability"] = probabilities

#phase2

url = "https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt"
response = requests.get(url)
arabic_stopwords = set(response.text.strip().split('\n'))

arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
stemmer=ISRIStemmer()

def remove_diacritics(text):
    return re.sub(arabic_diacritics, ' ', text)

def normalize_arabic(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^؀-ۿ ]+', ' ', text)
    return text

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

print("number of stopwords:", len(arabic_stopwords))
print("original text:", df["text"].iloc[0])
print("cleaned text:", df["cleaned_text"].iloc[0])
print("original length:", df['text'].str.len().mean())
print("cleaned length:", df['cleaned_text'].str.len().mean())
print("null values:", df['cleaned_text'].isna().sum())

def calculate_avg_word_length(texts):
    total_length = 0
    total_words = 0

    for text in texts:
        if pd.notna(text) and str(text).strip():
            words = str(text).split()
            for word in words:
                total_length += len(word)
                total_words += 1

    return total_length / total_words if total_words > 0 else 0

def calculate_avg_sentence_length(texts):
    total_words = 0
    total_sentences = 0

    for text in texts:
        if pd.notna(text) and str(text).strip():
            sentences = str(text).split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    words = sentence.split()
                    total_words += len(words)
                    total_sentences += 1

    return total_words / total_sentences if total_sentences > 0 else 0

def calculate_type_token_ratio(texts):
    all_words = []

    for text in texts:
        if pd.notna(text) and str(text).strip():
            words = str(text).split()
            all_words.extend(words)

    if len(all_words) == 0:
        return 0

    unique_words = set(all_words)
    return len(unique_words) / len(all_words)

human_texts = df[df['label'] == 'human']['cleaned_text']
ai_texts = df[df['label'] == 'ai']['cleaned_text']

human_avg = calculate_avg_word_length(human_texts)
ai_avg = calculate_avg_word_length(ai_texts)

human_sent_len = calculate_avg_sentence_length(human_texts)
ai_sent_len = calculate_avg_sentence_length(ai_texts)

human_ttr = calculate_type_token_ratio(human_texts)
ai_ttr = calculate_type_token_ratio(ai_texts)

def load_split_data(csv_path="/Users/lailaalmohaymid/PycharmProjects/data_mining/data/processed.csv"):
    df = pd.read_csv(csv_path)
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)
    return train_df, val_df, test_df

def get_datasets(csv_path="/Users/lailaalmohaymid/PycharmProjects/data_mining/data/processed.csv"):
    train_df, val_df, test_df = load_split_data(csv_path)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), analyzer="word")
    tfidf.fit(train_df["cleaned_text"])
    X_train_tfidf = tfidf.transform(train_df["cleaned_text"])
    X_val_tfidf = tfidf.transform(val_df["cleaned_text"])
    X_test_tfidf = tfidf.transform(test_df["cleaned_text"])
    num_cols = [
        'Total number of sentences',
        'Total number of paragraphs ',
        'Average number of S/P',
        'num_words_in_top50_embedding',
        'perplexity',
        'gpt2_output_probability'
    ]
    X_train_num = train_df[num_cols].values
    X_val_num = val_df[num_cols].values
    X_test_num = test_df[num_cols].values
    X_train = hstack([X_train_tfidf, X_train_num]).tocsr()
    X_val = hstack([X_val_tfidf, X_val_num]).tocsr()
    X_test = hstack([X_test_tfidf, X_test_num]).tocsr()
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    return X_train, X_val, X_test, y_train, y_val, y_test