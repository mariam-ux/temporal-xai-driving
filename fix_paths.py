import pandas as pd
import os

csv_path = 'data/driving_log.csv'

df = pd.read_csv(csv_path, header=None)

for col in [0, 1, 2]:
    df[col] = df[col].apply(lambda x: 'IMG/' + os.path.basename(str(x)))

df.to_csv(csv_path, index=False, header=False)

print("Paths updated successfully.")