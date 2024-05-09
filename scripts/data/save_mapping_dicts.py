import json
import os


def save_label_mapping(data_root):
    label_list = sorted(os.listdir(os.path.join(data_root, "train_audio")))
    label_id_list = list(range(len(label_list)))
    label2id = dict(zip(label_list, label_id_list))
    id2label = dict(zip(label_id_list, label_list))

    json_str = json.dumps(id2label)
    file_path = "id2label.json"
    with open(file_path, "w") as json_file:
        json_file.write(json_str)

    json_str = json.dumps(label2id)
    file_path = "label2id.json"
    with open(file_path, "w") as json_file:
        json_file.write(json_str)


if __name__ == "__main__":
    save_label_mapping("../data/bird-clef-2024/")
