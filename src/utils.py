import json
import torch

def save_to_json(obj, filename):
    jsonString = json.dumps(obj)
    jsonFile = open(filename, "w")
    jsonFile.write(jsonString)
    jsonFile.close()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)