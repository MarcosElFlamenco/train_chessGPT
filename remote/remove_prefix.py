

def remove_prefix_from_state_dict2(state_dict, prefix='_orig_mod.'):
    """
    Removes a specified prefix from the state_dict keys.

    Args:
        state_dict (dict): Original state_dict with prefixed keys.
        prefix (str): Prefix to remove from each key.

    Returns:
        dict: Updated state_dict with prefixes removed.
    """
    flag = False
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            flag = True
        else:
            new_key = key  # Keep the key as is
        new_state_dict[new_key] = value
    if flag:
        print('Unwanted prefixes were found in the checkpoint and have been removed.')
    return new_state_dict