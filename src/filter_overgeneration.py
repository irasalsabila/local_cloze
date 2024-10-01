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
