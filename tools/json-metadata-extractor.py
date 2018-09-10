# Converts individual JSON files into a usable CSV
import csv
import os
import json

def appendRecord(data_to_append):
    with open("extracted-data.csv", "a") as myfile:
        myfile.write(data_to_append)
        myfile.close()

directory = 'X://Datasets//isic//ISIC-images//UDA Combined'

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        path_to_file = directory + '//' + filename
        with open(path_to_file) as file:
            jsonfile = json.load(file)
            data_to_append = jsonfile["name"] + "," + jsonfile["_id"] + "," + jsonfile["meta"]["clinical"]["benign_malignant"]
            print(data_to_append)

            with open("C://Users//janie//Documents//GitHub//Project-Turing//tools//extracted-data.csv", "a+") as myfile:
                myfile.write(data_to_append + "\n")
                myfile.close()
