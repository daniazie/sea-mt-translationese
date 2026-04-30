import json
import os

def format_data(item: dict):
    messages = {
        "messages_foreignization": [{"role": "user", "content": f"Translate the following text to Malay:\n\n{item['source']}"}, {"role": "assistant", "content": item['foreignization']}],
        "messages_domestication": [{"role": "user", "content": f"Translate the following text to Malay:\n\n{item['source']}"}, {"role": "assistant", "content": item['domestication']}]
    } 

    item.update(messages)
    return item

dataset_dir = "data/translationese/synthetic/enms"
for model_dir in os.listdir(dataset_dir):
    data_dir = f"{dataset_dir}/{model_dir}"
    data_files = os.listdir(data_dir)
    for data_file in data_files:
        with open(f"{data_dir}/{data_file}", "r") as file:
            data = []
            for line in file.readlines():
                data.append(json.loads(line))
                
        data = list(map(lambda item: format_data(item), data))
        with open(f"{data_dir}/{data_file}", "w") as file:
            for item in data:
                json.dump(item, file)
                file.write('\n')