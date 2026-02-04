import numpy as np
import pandas as pd
import pickle

svc = pickle.load(open("model/svc.pkl", "rb"))

symptoms_dict = {...}      # same dict
diseases_list = {...}      # same dict

description = pd.read_csv("data/description.csv")
precautions = pd.read_csv("data/precautions_df.csv")
medications = pd.read_csv("data/medications.csv")
diets = pd.read_csv("data/diets.csv")
workout = pd.read_csv("data/workout_df.csv")

def get_predicted_value(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        input_vector[symptoms_dict[s]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def helper(disease):
    desc = description[description["Disease"] == disease]["Description"].values[0]
    pre = precautions[precautions["Disease"] == disease].values.tolist()
    med = medications[medications["Disease"] == disease]["Medication"].tolist()
    die = diets[diets["Disease"] == disease]["Diet"].tolist()
    wrk = workout[workout["disease"] == disease]["workout"].tolist()
    return desc, pre, med, die, wrk
