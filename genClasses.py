import pandas as pd
import numpy as np

df = pd.read_csv('train_1_info_30_percent.csv')

df.sort_values(by=['date'], inplace=True)
length = len(df)
num_classes = 5
indicies = []
for i in range(1,num_classes):
	indicies.append(int(i * length / num_classes))
same_values = False
for i in range(num_classes - 2):
	if indicies[i] == indicies[i + 1]:
		same_values = True
		break
if (same_values == True):
	print("UH OH")
dfl = []
dates = []
for i in range(num_classes - 1):
	dfl.append(df.iloc[indicies[i]])
	dates.append(dfl[i]['date'])
print(indicies)
print(dates)
print(dates[1:-1])

df['class'] = np.zeros
date = dates[0]
df['class'].loc[df['date'] <= date] = 0

count = 1
previous_date = date
for date in dates[1:]:
	df['class'].loc[(df['date'] > previous_date) & (df['date'] <= date)] = count
	previous_date = date
	count += 1
	
date = dates[-1]
df['class'].loc[df['date'] > date] = count
df.drop(columns=['date'], inplace=True)
print(df)
#df['class'] = df['class'].astype(int)
df.to_csv('train_1_5_classes_30_percent.csv', index = False, header=True)