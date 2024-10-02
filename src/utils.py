import pandas as pd

def balance_topic_distribution(df, max_rows=1000, rows_per_topic=None):
    """
    Balance the DataFrame by topic and save to a new CSV file.

    Parameters:
    - df: The DataFrame to be balanced.
    - file_path: Path to save the balanced DataFrame.
    - max_rows: Maximum number of rows for the balanced DataFrame.
    - rows_per_topic: Number of rows per topic. If None, calculated based on max_rows.

    Returns:
    - balanced_df: The balanced DataFrame.
    """
    unique_topics = df['topic'].unique()
    if rows_per_topic is None:
        rows_per_topic = max_rows // len(unique_topics)  # Adjust this as needed

    # Create a balanced DataFrame
    balanced_df = pd.DataFrame(columns=df.columns)
    remaining_rows_needed = max_rows

    # Dictionary to track the number of rows sampled from each topic
    sampled_counts = {}

    for topic in unique_topics:
        # Filter rows for the current topic
        subset = df[df['topic'] == topic]

        # Sample rows to balance the data
        sampled_subset = subset.sample(
            min(len(subset), rows_per_topic),
            random_state=42
        )

        # Update the sampled counts
        sampled_counts[topic] = len(sampled_subset)

        # Append the sampled subset to the balanced DataFrame
        balanced_df = pd.concat([balanced_df, sampled_subset], ignore_index=True)

        # Decrease the number of remaining rows needed
        remaining_rows_needed -= len(sampled_subset)

    # If we still need more rows, add them from the remaining available rows
    if remaining_rows_needed > 0:
        # Sort topics by the number of available rows in descending order
        topic_counts = df['topic'].value_counts()
        for topic in topic_counts.index:
            if remaining_rows_needed <= 0:
                break

            # Filter rows for the current topic
            subset = df[df['topic'] == topic]

            # Calculate the number of additional rows to sample
            additional_rows = min(remaining_rows_needed, len(subset) - sampled_counts.get(topic, 0))

            if additional_rows > 0:
                sampled_subset = subset.sample(
                    additional_rows,
                    random_state=42
                )

                # Append the additional sampled rows to the balanced DataFrame
                balanced_df = pd.concat([balanced_df, sampled_subset], ignore_index=True)

                # Update the remaining rows needed
                remaining_rows_needed -= additional_rows

    # Ensure that the balanced DataFrame has exactly max_rows
    balanced_df = balanced_df.sample(n=max_rows, random_state=42)
    return balanced_df