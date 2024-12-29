import json

def save_dict_to_json(data: dict, filename: str):
    """
    Saves all key-value pairs from a dictionary to a JSON file, excluding the key 'context'.
    
    Args:
        data (dict): The dictionary containing the data to save.
        filename (str): The name of the JSON file to create.
    """
    filtered_data = {key: value for key, value in data.items() if key != 'context'}
    with open(filename, 'w') as json_file:
        json.dump(filtered_data, json_file, indent=4)