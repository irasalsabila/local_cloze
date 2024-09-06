def remove_duplicates(input_list):
    seen = []
    unique_list = []

    for item in input_list:
        # Convert the dictionary to a frozenset of its items for hashing
        item_tuple = frozenset(item.items())
        if item_tuple not in seen:
            seen.append(item_tuple)
            unique_list.append(item)

    return unique_list

result = remove_duplicates(responses_final)