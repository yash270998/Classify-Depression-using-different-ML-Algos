import pandas as pd
import os
import numpy as np

ConditionGroupFileNames = os.listdir('data/condition')
ControlGroupFileNames = os.listdir('data/control')

X_condition = []

for fileName in ConditionGroupFileNames:
    df = pd.read_csv('data/condition/'+str(fileName))
    dates = df['date'].unique()
    activityLevelsPerDay = []
    for date in dates:
        if len(df[df['date']==date]) == 1440:
            temp = pd.DataFrame(df[df['date']==date]).drop(columns=['timestamp','date'])
            activityLevelsPerDay.append(temp)
    for dailyActivityLevel in activityLevelsPerDay:
        activityVector = np.array(dailyActivityLevel["activity"])
        if len(activityVector) == 1440:
            X_condition.append(activityVector)

X_control = []

for fileName in ControlGroupFileNames:
    df = pd.read_csv('data/control/'+str(fileName))
    dates = df['date'].unique()
    activityLevelsPerDay = []
    for date in dates:
        if len(df[df['date']==date]) == 1440:
            temp = pd.DataFrame(df[df['date']==date]).drop(columns=['timestamp','date'])
            activityLevelsPerDay.append(temp)
    for dailyActivityLevel in activityLevelsPerDay:
        activityVector = np.array(dailyActivityLevel["activity"])
        if len(activityVector) == 1440:
            X_control.append(activityVector)
    
import matplotlib.pyplot as plt
import seaborn as sns

condition_sum_vector = X_condition[0]
for x in range(1, len(X_condition)):
    condition_sum_vector += X_condition[x]
condition_avg_vector = condition_sum_vector / len(X_condition)

plt.figure(figsize=(20, 10))
plt.plot(condition_avg_vector)
plt.title('Condition Group Average Daily Activity Time-Series')
plt.ylabel('Activity Level')
plt.xlabel('Minute of the day')
plt.grid(False)
# plt.show()
plt.savefig('Condition Group Average Daily Activity Time-Series.png')

control_sum_vector = X_control[0]
for x in range(1, len(X_control)):
    control_sum_vector += X_control[x]
control_sum_vector = control_sum_vector / len(X_control)

plt.figure(figsize=(20, 10))
plt.plot(control_sum_vector)
plt.title('Control Group Average Daily Activity Time-Series')
plt.ylabel('Activity Level')
plt.xlabel('Minute of the day')
plt.grid(False)
# plt.show()
plt.savefig('Control Group Average Daily Activity Time-Series.png')

plt.figure(figsize=(10, 5))
plt.plot(control_sum_vector, label='control')
plt.plot(condition_avg_vector, label='condition')
plt.title('Control vs Condition Average Daily Activity Time-Series')
plt.ylabel('Activity Level')
plt.xlabel('Minute of the day')
plt.legend()
plt.grid(False)
plt.savefig('Control vs Condition Average Daily Activity Time-Series.png')
# plt.show()

