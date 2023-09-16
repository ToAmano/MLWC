
import csv
with open('liquids_oligomers.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)


import pandas as pd
df = pd.read_csv('liquids_oligomers.csv')
print(df)
