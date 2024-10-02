import pandas as pd


def filter_heuristics(df, df_icl):
    print("Initial amount ", df.shape)
    df[["sentence_1", "sentence_2", "sentence_3", "sentence_4"]] = (
        df["premise"].apply(split_premise).apply(pd.Series)
    )
    df.drop_duplicates(inplace=True)
    df.drop(columns=["premise"], inplace=True)    
    common_columns = df.columns.intersection(df_icl.columns)
    df = df[~df[common_columns].apply(tuple, axis=1).isin(df_icl[common_columns].apply(tuple, axis=1))]# remove icl duplicates
    print("Remove duplication (including ICL echoing): ", df.shape)
    df = remove_short_samples(df)
    df = df[
        df.apply(lambda row: len(set(row)) == len(row), axis=1)
    ]  # make sure that each row have a unique column value (story premises are not repeting, incorrect ending and correct ending are differents, and endings does not echo story premises)
    validity_df = df.map(is_valid_sentence)
    df = df[validity_df.all(axis=1)]
    df = df[df['correct_ending'].apply(lambda x: is_single_sentence(x))]
    # 6. Remove rows where incorrect_ending is None or has more than one sentence
    df = df[df['incorrect_ending'].apply(lambda x: is_single_sentence(x))]
    # Filter rows where all cells in the row are True (i.e., all cells contain a valid sentence)
    print("Remove broken examples:", df.shape)
    df = df[
        df.apply(lambda row: remove_samples_containing_instruction_phrases(row), axis=1)
    ]
    
    return df


def split_premise(premise):
    if isinstance(premise, str):  # Ensure premise is a string
        # Split based on the period (.)
        sentences = premise.split(".")

        # Remove empty sentences or whitespace-only sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # If there are less than 4 sentences, pad with empty strings
        while len(sentences) < 4:
            sentences.append("")

        # Return the first four sentences
        return sentences[:4]
    else:
        # Return empty values if premise is not a valid string
        return ["", "", "", ""]


def remove_short_samples(df):
    return df[
        (df["sentence_1"].str.len() > 5)
        & (df["sentence_2"].str.len() > 5)
        & (df["sentence_3"].str.len() > 5)
        & (df["sentence_4"].str.len() > 5)
        & (df["correct_ending"].str.len() > 5)
        & (df["incorrect_ending"].str.len() > 5)
    ]

def remove_samples_containing_instruction_phrases(row):
    INSTRUCTION_PHRASES = [
        "please generate" "write several triplets",
        "story premises consisting",
        "include sundanese cultural values",
        "include javanese cultural values",
        "generate triplets",
    ]
    for col in row:
        # Check if any prompt is a substring of the column value
        if any(phrase in str(col) for phrase in INSTRUCTION_PHRASES):
            return False
    return True

import re
# Function to check if a string is a valid sentence (not just punctuation or quotes)
def is_valid_sentence(sentence):
    return bool(re.search(r'\w', sentence))  # Returns True if there's at least one word character


# Function to check if an ending has more than one sentence
def is_single_sentence(ending):
    if pd.isna(ending):  # Check if the ending is NaN
        return False
    sentences = ending.split('.')
    # Strip whitespace and filter out empty sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return len(sentences) == 1
