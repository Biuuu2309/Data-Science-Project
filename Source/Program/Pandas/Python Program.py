# %% [markdown]
# Import data

# %%
# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("kamilpytlak/personal-key-indicators-of-heart-disease")

# print("Path to dataset files:", path)

# %% [markdown]
# Dia chi luu file csv
# 
# C:\Users\minhv\.cache\kagglehub\datasets\asaniczka\tmdb-movies-dataset-2023-930k-movies\versions\560

# %% [markdown]
# Import thu vien model can thiet

# %%
import json
import pandas as pd
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from joypy import joyplot
from matplotlib.cm import viridis
import plotly.express as px
import squarify
from numpy import float64

# %%
DATA = pd.read_csv(r"E:\Data Science Project\Source\Data\heart_2022_with_nans.csv")
DATA = pd.DataFrame(DATA)
DATA.insert(column= 'PersonID', loc= 1, value=['Person' + str(i) for i in range(len(DATA))])
DATA1 = DATA
DATA2 = DATA

# %%
DATA1["PhysicalHealthDays"].fillna(min(DATA1["PhysicalHealthDays"].mean(), DATA1["PhysicalHealthDays"].mode()[0], DATA1["PhysicalHealthDays"].median()), inplace=True)
DATA1["MentalHealthDays"].fillna(min(DATA1["MentalHealthDays"].mean(), DATA1["MentalHealthDays"].mode()[0], DATA1["MentalHealthDays"].median()), inplace=True)
DATA1["SleepHours"].fillna(min(DATA1["SleepHours"].mean(), DATA1["SleepHours"].mode()[0], DATA1["SleepHours"].median()), inplace=True)
DATA1["HeightInMeters"].fillna(min(DATA1["HeightInMeters"].mean(), DATA1["HeightInMeters"].mode()[0], DATA1["HeightInMeters"].median()), inplace=True)
DATA1["WeightInKilograms"].fillna(min(DATA1["WeightInKilograms"].mean(), DATA1["WeightInKilograms"].mode()[0], DATA1["WeightInKilograms"].median()), inplace=True)
DATA1["BMI"].fillna(min(DATA1["BMI"].mean(), DATA1["BMI"].mode()[0], DATA1["BMI"].median()), inplace=True)
DATA1


# %%
# group_count = 500
# n = len(DATA1)
# groups = np.array_split(DATA1.index, group_count)

# for group in groups:
#     mean_val = DATA1.loc[group, 'WeightInKilograms'].mean()
#     DATA1.loc[group, 'WeightInKilograms'] = mean_val

# DATA1['WeightInKilograms'] = np.round(DATA1['WeightInKilograms'], 2)
# DATA1

# %%
# group_count = 500
# for i in range(group_count):
#     if i == group_count - 1:
#         for j in range(len(DATA1[int(len(DATA1["WeightInKilograms"]) / group_count) * (group_count - 1) : len(DATA1["WeightInKilograms"])])):
#             DATA1.loc[j, 'WeightInKilograms'] = DATA1["WeightInKilograms"][int(len(DATA1["WeightInKilograms"]) / group_count) * (group_count - 1) : len(DATA1["WeightInKilograms"])].mean()
#     else:
#         for j in range(len(DATA1[int(len(DATA1["WeightInKilograms"]) / group_count) * i : int(len(DATA1["WeightInKilograms"]) / group_count) * (i + 1)])):
#             DATA1.loc[j, 'WeightInKilograms'] = DATA1["WeightInKilograms"][int(len(DATA1["WeightInKilograms"]) / group_count) * i : int(len(DATA1["WeightInKilograms"]) / group_count) * (i + 1)].mean()
# DATA1

# %%
DATA1_2 = DATA2

DATA1_2_1 = (DATA1_2['PhysicalHealthDays']**2).sum()
DATA1_2_2 = (DATA1_2['MentalHealthDays']**2).sum()
DATA1_2_3 = (DATA1_2['SleepHours']**2).sum()
DATA1_2_4 = (DATA1_2['HeightInMeters']**2).sum()
DATA1_2_5 = (DATA1_2['WeightInKilograms']**2).sum()
DATA1_2_6 = (DATA1_2['BMI']**2).sum()


a1 = DATA1_2_1 / len(DATA1_2) - DATA1_2['PhysicalHealthDays'].mean()**2
a2 = DATA1_2_2 / len(DATA1_2) - DATA1_2['MentalHealthDays'].mean()**2
a3 = DATA1_2_3 / len(DATA1_2) - DATA1_2['SleepHours'].mean()**2
a4 = DATA1_2_4 / len(DATA1_2) - DATA1_2['HeightInMeters'].mean()**2
a5 = DATA1_2_5 / len(DATA1_2) - DATA1_2['WeightInKilograms'].mean()**2
a6 = DATA1_2_6 / len(DATA1_2) - DATA1_2['BMI'].mean()**2

#variance = sum((DF['views'] - DF['views'].mean())**2) / len(DF)

#std_views = variance**0.5
print(a1**0.5, a2**0.5, a3**0.5, a4**0.5, a5**0.5, a6**0.5)

# %%
DATA1.sort_values(by='PhysicalHealthDays', ascending=True)
Q1 = DATA1['PhysicalHealthDays'].quantile(0.25)
Q3 = DATA1['PhysicalHealthDays'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = max(0, Q1 - 1.5 * IQR)
outliers = DATA1[
    (DATA1['PhysicalHealthDays'] < lower_bound) |
    (DATA1['PhysicalHealthDays'] > Q3 + 1.5 * IQR)
]
outliers

# %%
DATA1 = pd.DataFrame(DATA1)

State_mapping = {i: j + 1 for j, i in enumerate(DATA1["State"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_State_mapping = {v: k for k, v in State_mapping.items()}
DATA1["State"] = DATA1["State"].map(State_mapping)

Sex_mapping = {"Male": 1, "Female": 0}
DATA1["Sex"] = DATA1["Sex"].map(Sex_mapping)
reverse_Sex_mapping = {v: k for k, v in Sex_mapping.items()}

GeneralHealth_mapping = {i: j + 1 for j, i in enumerate(DATA1["GeneralHealth"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_GeneralHealth_mapping = {v: k for k, v in GeneralHealth_mapping.items()}
DATA1["GeneralHealth"] = DATA1["GeneralHealth"].map(GeneralHealth_mapping)

LastCheckupTime_mapping = {i: j + 1 for j, i in enumerate(DATA1["LastCheckupTime"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_LastCheckupTime_mapping = {v: k for k, v in LastCheckupTime_mapping.items()}
DATA1["LastCheckupTime"] = DATA1["LastCheckupTime"].map(LastCheckupTime_mapping)

RemovedTeeth_mapping = {i: j + 1 for j, i in enumerate(DATA1["RemovedTeeth"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_RemovedTeeth_mapping = {v: k for k, v in RemovedTeeth_mapping.items()}
DATA1["RemovedTeeth"] = DATA1["RemovedTeeth"].map(RemovedTeeth_mapping)

SmokerStatus_mapping = {i: j + 1 for j, i in enumerate(DATA1["SmokerStatus"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_SmokerStatus_mapping = {v: k for k, v in SmokerStatus_mapping.items()}
DATA1["SmokerStatus"] = DATA1["SmokerStatus"].map(SmokerStatus_mapping)

ECigaretteUsage_mapping = {i: j + 1 for j, i in enumerate(DATA1["ECigaretteUsage"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_ECigaretteUsage_mapping = {v: k for k, v in ECigaretteUsage_mapping.items()}
DATA1["ECigaretteUsage"] = DATA1["ECigaretteUsage"].map(ECigaretteUsage_mapping)

RaceEthnicityCategory_mapping = {i: j + 1 for j, i in enumerate(DATA1["RaceEthnicityCategory"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_RaceEthnicityCategory_mapping = {v: k for k, v in RaceEthnicityCategory_mapping.items()}
DATA1["RaceEthnicityCategory"] = DATA1["RaceEthnicityCategory"].map(RaceEthnicityCategory_mapping)

AgeCategory_mapping = {i: j + 1 for j, i in enumerate(DATA1["AgeCategory"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_AgeCategory_mapping = {v: k for k, v in AgeCategory_mapping.items()}
DATA1["AgeCategory"] = DATA1["AgeCategory"].map(AgeCategory_mapping)

TetanusLast10Tdap_mapping = {i: j + 1 for j, i in enumerate(DATA1["TetanusLast10Tdap"].drop_duplicates().sort_values(ascending=True), start=0)}
reverse_TetanusLast10Tdap_mapping = {v: k for k, v in TetanusLast10Tdap_mapping.items()}
DATA1["TetanusLast10Tdap"] = DATA1["TetanusLast10Tdap"].map(TetanusLast10Tdap_mapping)

lst = ['PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos']
for i in lst:
    DATA1[i] = DATA1[i].map({'Yes': 1, 'No': 0})
DATA1

# %%
DATA_VI = DATA

# %%
country_order = DATA_VI.groupby('AgeCategory')['WeightInKilograms'].median().sort_values().index
plt.figure(figsize=(12, 6))
sns.boxplot(x='AgeCategory', y='WeightInKilograms', data=DATA_VI, order=country_order, palette='Set2')
plt.title("Boxplot Chart: About AgeCategory and WeightInKilograms Sorted by Median", fontsize=14)
plt.xlabel("Age Category", fontsize=12)
plt.ylabel("Weight In Kilograms", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# df = pd.read_csv("heart_disease_indicators.csv")

x = DATA_VI['PhysicalHealthDays'].dropna()
y = DATA_VI['MentalHealthDays'].dropna()

min_len = min(len(x), len(y))
x = x.sample(min_len, random_state=1).sort_values().reset_index(drop=True)
y = y.sample(min_len, random_state=1).sort_values().reset_index(drop=True)

plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o')
plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--')  # đường 45 độ

plt.title('QQ Plot: PhysicalHealthDays vs MentalHealthDays')
plt.xlabel('Quantiles of PhysicalHealthDays')
plt.ylabel('Quantiles of MentalHealthDays')
plt.grid(True)
plt.show()


# %%
plt.figure(figsize=(8, 6))
plt.scatter(DATA_VI['HeightInMeters'], DATA_VI['WeightInKilograms'], alpha=0.1, color='blue', s=10)
plt.title('Scatter Plot: HeightInMeters vs WeightInKilograms')
plt.xlabel('HeightInMeters')
plt.ylabel('WeightInKilograms')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
scatter = plt.scatter(DATA_VI['HeightInMeters'], DATA_VI['WeightInKilograms'], c=DATA_VI['BMI'], cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='BMI')  
plt.title('Scatter Plot: Height vs Weight (Colored by BMI)')
plt.xlabel('HeightInMeters')
plt.ylabel('WeightInKilograms')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
country_order = DATA_VI.groupby('AgeCategory')['HeightInMeters'].median().sort_values().index
plt.figure(figsize=(12, 6))
sns.boxplot(x='AgeCategory', y='HeightInMeters', data=DATA_VI, order=country_order, palette='Set2')
plt.title("Boxplot Chart: About AgeCategory and HeightInMeters Sorted by Median", fontsize=14)
plt.xlabel("Age Category", fontsize=12)
plt.ylabel("Height In Kilograms", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
country_order = DATA_VI.groupby('GeneralHealth')['SleepHours'].median().sort_values().index
plt.figure(figsize=(12, 6))
sns.boxplot(x='GeneralHealth', y='SleepHours', data=DATA_VI, order=country_order, palette='Set2')
plt.title("Boxplot Chart: About GeneralHealth and SleepHours Sorted by Median", fontsize=14)
plt.xlabel("General Health", fontsize=12)
plt.ylabel("SleepHours", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
import seaborn as sns

sns.histplot(data=DATA_VI, x='BMI', hue='HadHeartAttack', multiple='stack', bins=30)
plt.title('Histogram of BMI grouped by Heart Attack Status')
plt.show()

# %%
plt.figure(figsize=(8, 6))
plt.hist(DATA_VI['HeightInMeters'].dropna(), bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.title('Histogram of HeightInMeters')
plt.xlabel('HeightInMeters (Body Mass Index)')
plt.ylabel('Number of People')
plt.grid(True)
plt.show()

# %%
df_grouped = DATA1['GeneralHealth'].value_counts().reset_index()
df_grouped.columns = ['GeneralHealth', 'count']
df_grouped['fraction'] = df_grouped['count'] / df_grouped['count'].sum()
df_grouped['percentage'] = round(df_grouped['fraction'] * 100, 1)
df_grouped['label'] = df_grouped['GeneralHealth'].map(reverse_GeneralHealth_mapping) + ": " + df_grouped['percentage'].astype(str) + "%"

plt.figure(figsize=(8, 8))
plt.pie(df_grouped['fraction'], labels=df_grouped['label'], autopct='%1.1f%%',
        startangle=90, colors=plt.cm.Paired.colors)
plt.title("Pie Chart: GeneralHealth percent by count", fontsize=14)
plt.axis('equal')  
plt.show()


# %%
df_grouped = DATA1['LastCheckupTime'].value_counts().reset_index()
df_grouped.columns = ['LastCheckupTime', 'count']
df_grouped['fraction'] = df_grouped['count'] / df_grouped['count'].sum()
df_grouped['percentage'] = round(df_grouped['fraction'] * 100, 1)
df_grouped['label'] = df_grouped['LastCheckupTime'].map(reverse_LastCheckupTime_mapping) + ": " + df_grouped['percentage'].astype(str) + "%"

plt.figure(figsize=(8, 8))
plt.pie(df_grouped['fraction'], labels=df_grouped['label'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, wedgeprops={'width': 0.6})
plt.title("Doughnut Chart: Views by LastCheckupTime", fontsize=14)
plt.axis('equal')
plt.show()

# %%


df_grouped = DATA1.groupby('AgeCategory', as_index=False)['ECigaretteUsage'].count()
labels = [reverse_AgeCategory_mapping.get(x, str(x)) for x in df_grouped['AgeCategory']]
plt.figure(figsize=(10, 6))
plt.plot(df_grouped['AgeCategory'], df_grouped['ECigaretteUsage'], color='red', marker='o', markersize=5)
plt.xticks(df_grouped['AgeCategory'], labels, rotation=45, ha='right')
plt.title("Line chart: AgeCategory by ECigaretteUsage", fontsize=14)
plt.xlabel("AgeCategory", fontsize=12)
plt.ylabel("ECigaretteUsage Count", fontsize=12)
plt.tight_layout()
plt.show()

# %%
df_grouped1 = DATA1.groupby('Sex', as_index=False)['SmokerStatus'].count()
labelx = [reverse_Sex_mapping.get(x, str(x)) for x in df_grouped1['Sex']]
df_grouped2 = DATA1.groupby('SmokerStatus', as_index=False)['Sex'].count()
labely = [reverse_SmokerStatus_mapping.get(x, str(x)) for x in df_grouped2['SmokerStatus']]
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sex', y='SmokerStatus', data=DATA1, palette='Set2')
plt.title("Violin Chart: SmokerStatus with Sex", fontsize=14)
plt.xlabel("SmokerStatus", fontsize=12)
plt.ylabel("Sex", fontsize=12)
plt.xticks(df_grouped1['Sex'], labelx, rotation=45, ha='right')
plt.yticks(df_grouped2['SmokerStatus'], labely)
plt.tight_layout()
plt.show()

# %%
df_grouped1 = DATA1.groupby('GeneralHealth', as_index=False)['SmokerStatus'].count()
labelx = [reverse_GeneralHealth_mapping.get(x, str(x)) for x in df_grouped1['GeneralHealth']]

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=DATA1,
    x='GeneralHealth',
    hue='SmokerStatus',
    common_norm=False,
    fill=True,
    alpha=0.5,
    palette='Set2'
)
plt.title("Density Chart: GeneralHealth of SmokerStatus", fontsize=14)
plt.xlabel("GeneralHealth", fontsize=12)
plt.xticks(df_grouped1['GeneralHealth'], labelx, rotation=45, ha='right')
plt.ylabel("Density", fontsize=12)
plt.tight_layout()
plt.show()

# %%
data_grouped = [DATA1[DATA1['AgeCategory'] == day]['GeneralHealth'].dropna() for day in DATA1['AgeCategory'].unique()]
labels = list(DATA1['AgeCategory'].unique())

joyplot(
    pd.DataFrame(dict(zip(labels, data_grouped))),
    fade=True,
    figsize=(10, 6),
    title="Ridgeline Plot: Distribution of GeneralHealth by AgeCategory",
    colormap=plt.cm.Paired
)

xticks = list(GeneralHealth_mapping.values())  # [1, 2, 3, 4, 5]
xtick_labels = [reverse_GeneralHealth_mapping[i] for i in xticks]

plt.xticks(ticks=xticks, labels=xtick_labels, rotation=45, ha='right')
plt.xlabel("General Health Category", fontsize=12)
plt.ylabel("Age Category", fontsize=12)
plt.tight_layout()
plt.show()


# %%
data = DATA1[['PhysicalHealthDays', 'HeightInMeters', 'WeightInKilograms', 'BMI']]
sns.pairplot(data, kind='scatter', plot_kws={'alpha': 0.7}, diag_kind='hist', markers='o', hue=None, )
plt.suptitle("Body Stat with PhysicalHealthDays", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# %%
data = (DATA1.groupby(['Sex', 'HadHeartAttack']).size().reset_index(name='count'))
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Sex', y='count', hue='HadHeartAttack', palette='viridis')
plt.title("Stacked Barplot: Counts Sex by HadHeartAttack", fontsize=16)
plt.xlabel("Sex", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="HadHeartAttack", loc="upper right")
plt.tight_layout()
plt.show()

# %%
data = (DATA1.groupby('State')['HadHeartAttack'].count().reset_index(name='cnt_HHA').sort_values(by='cnt_HHA', ascending=False))
data['State'] = pd.Categorical(data['State'], categories=data['State'], ordered=True)
angles = np.linspace(0, 2 * np.pi, len(data), endpoint=False)  
data['angle'] = angles 
colors = viridis(np.linspace(0, 1, len(data)))
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
bars = ax.bar(data['angle'], data['cnt_HHA'], color=colors, edgecolor='black', width=0.4, align='center')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1) 
ax.set_xticks(data['angle'])
ax.set_xticklabels(data['State'])
ax.set_yticks([])  
ax.set_title("Circular Barplot: State by HadHeartAttack Count", va='bottom', fontsize=16)
plt.tight_layout()
plt.show()

# %%
data = (DATA1.groupby(['State', 'GeneralHealth'])['HadHeartAttack'].count().reset_index(name='total_HadHeartAttack'))
heatmap_data = data.pivot(index='GeneralHealth', columns='State', values='total_HadHeartAttack')
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, fmt=".0f", cmap="Blues", linewidths=.5, cbar_kws={'label': 'Count HadHeartAttack'})
plt.title("Heatmap: Count HadHeartAttack by State and GeneralHealth", fontsize=16)
plt.xlabel("State", fontsize=12)
plt.ylabel("GeneralHealth", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
DATA1["SmokerStatus_Label"] = DATA1["SmokerStatus"].map(reverse_SmokerStatus_mapping)
data = (DATA1.groupby(['SmokerStatus_Label', 'AgeCategory'])['HadCOPD'].count().reset_index())
data.columns = ['SmokerStatus', 'AgeCategory', 'HadCOPD']

data['text'] = ("Smoker Status: " + data['SmokerStatus'].astype(str)+ "<br>Age Category: " + data['AgeCategory'].astype(str)+ "<br>Count: " + data['HadCOPD'].astype(str))

fig = px.scatter(
    data,
    x="AgeCategory",
    y="HadCOPD",
    size="HadCOPD",
    color="SmokerStatus",
    hover_name="SmokerStatus",
    hover_data={'text': True},
    title="Bubble Chart: COPD Cases by Smoker Status and Age Category",
    labels={"AgeCategory": "Age Category", "HadCOPD": "Number of Cases"},
    size_max=20,
    template="plotly_white"
)

fig.update_traces(marker=dict(opacity=0.7))
fig.update_xaxes(tickangle=45)
fig.update_layout(legend_title="Smoker Status")
fig.show()



# %%
grouped = DATA1.groupby(['State', 'AgeCategory', 'GeneralHealth']).size().reset_index(name='Count')
fig = px.treemap(
    grouped,
    path=['State', 'AgeCategory', 'GeneralHealth'],
    values='Count',
    title='Tree Map of General Health by State and Age Category'
)
fig.show()


# %%
DATA_AN = DATA_VI
DATA_AN1 = DATA_AN[['PersonID', 'Sex', 'SmokerStatus', 'HadHeartAttack', 'HadAngina', 'AlcoholDrinkers', 'PhysicalActivities', 'HeightInMeters', 'WeightInKilograms', 'SleepHours', 'GeneralHealth']].sample(n=15)
DATA_AN1["WeightInKilograms"].fillna(DATA_AN1["WeightInKilograms"].mean(), inplace=True)
DATA_AN1

# %%
from numpy import int64

arr1 = np.zeros((15, 15), dtype=int64)
for i in range(len(DATA_AN1)):
    for j in range(i):
        a1 = DATA_AN1.iloc[i]['Sex']
        a2 = DATA_AN1.iloc[j]['Sex']
        arr1[i, j] = 1 if a1 != a2 else 0
arr_sex = arr1
print(arr_sex)

arr2 = np.zeros((15, 15), dtype=int64)
for i in range(len(DATA_AN1)):
    for j in range(i):
        a1 = DATA_AN1.iloc[i]['SmokerStatus']
        a2 = DATA_AN1.iloc[j]['SmokerStatus']
        arr2[i, j] = 1 if a1 != a2 else 0
arr_ss = arr2
print(arr_ss)

# %%
DATA_BA1 = DATA_AN1[['PersonID', 'HadHeartAttack', 'HadAngina', 'AlcoholDrinkers', 'PhysicalActivities']]

arr = np.array([])
dict_ba = {}

for col in ['HadHeartAttack', 'HadAngina', 'AlcoholDrinkers', 'PhysicalActivities']:
    q = r = s = t = 0
    n = len(DATA_BA1)
    for i in range(n):
        for j in range(i+1, n):
            val_i = DATA_BA1.iloc[i][col]
            val_j = DATA_BA1.iloc[j][col]
            if val_i == 1 and val_j == 1:
                q += 1
            elif val_i == 1 and val_j == 0:
                r += 1
            elif val_i == 0 and val_j == 1:
                s += 1
            else:
                t += 1
            arr = np.append(arr, [col, i, j, q, r, s, t])
            dict_ba.update({'d' + str(i) + '_' + str(j): None})
            q = r = s = t = 0

n = len(arr)
arr1 = arr[0:n//4]
arr2 = arr[n//4:n//2]
arr3 = arr[n//2:3*n//4]
arr4 = arr[3*n//4:]

df1, df2, df3, df4 = [], [], [], []  

i = 3  
while i < len(arr1):
    df1.extend(arr1[i:i+4])
    df2.extend(arr2[i:i+4])
    df3.extend(arr3[i:i+4])
    df4.extend(arr4[i:i+4])
    i += 7 

df1 = np.array(df1, dtype=float)
df2 = np.array(df2, dtype=float)
df3 = np.array(df3, dtype=float)
df4 = np.array(df4, dtype=float)
arr_sum = df1 + df2 + df3 + df4

new_dict = {}

keys = list(dict_ba.keys())
index = 0

for k in keys:
    if index + 2 < len(arr_sum):
        try:
            a = float(arr_sum[index])
            b = float(arr_sum[index + 1])
            c = float(arr_sum[index + 2])
            new_dict[k] = (b + c) / (a + b + c) if (a + b + c) != 0 else 0
        except (ValueError, ZeroDivisionError):
            new_dict[k] = 0
        index += 4
    else:
        break

dict_ba.update(new_dict)


# %%

n = len(DATA_BA1)
arr_ba_lower = np.zeros((n, n), dtype=float)
for key, value in dict_ba.items():
    try:
        indices = key[1:].split('_')
        i = int(indices[0])
        j = int(indices[1])
        if i > j:
            arr_ba_lower[i, j] = value
        elif j > i:
            arr_ba_lower[j, i] = value  
    except:
        continue
print(np.round(arr_ba_lower, 2))


# %%
from numpy import float64


dict_gh = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5}
dict_cal = {1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1}

DATA_BA2 = DATA_AN1['GeneralHealth'].map(dict_gh)

n = len(DATA_BA2)
arr_gh = np.zeros((n, n), dtype=float64)

for i in range(n):
    for j in range(i+1, n):
        val_i = dict_cal.get(DATA_BA2.iloc[i], np.nan)
        val_j = dict_cal.get(DATA_BA2.iloc[j], np.nan)
        if not np.isnan(val_i) and not np.isnan(val_j):
            distance = abs(val_i - val_j)
            arr_gh[i, j] = distance
            arr_gh[j, i] = distance  

n = arr_gh.shape[0]

arr_gh_lower = np.zeros_like(arr_gh)

for i in range(n):
    for j in range(i + 1):  
        arr_gh_lower[i, j] = arr_gh[i, j]
print(arr_gh_lower)

# %%
from numpy import float64

DATA_BA3 = DATA_AN1['HeightInMeters']
n = len(DATA_BA3)
arr_htm = np.zeros((n, n), dtype=float64)

min_ht = DATA_BA3.min()
max_ht = DATA_BA3.max()

for i in range(n):
    for j in range(i + 1, n):
        val_i = (DATA_BA3.iloc[i] - min_ht) / (max_ht - min_ht)
        val_j = (DATA_BA3.iloc[j] - min_ht) / (max_ht - min_ht)
        if not np.isnan(val_i) and not np.isnan(val_j):
            distance = abs(val_i - val_j)
            arr_htm[i, j] = distance
            arr_htm[j, i] = distance

arr_htm_lower = np.tril(arr_htm)

print(np.round(arr_htm_lower, 2))


# %%
from numpy import float64

DATA_BA4 = DATA_AN1['WeightInKilograms']
n = len(DATA_BA4)
arr_wik = np.zeros((n, n), dtype=float64)

min_ht = DATA_BA4.min()
max_ht = DATA_BA4.max()

for i in range(n):
    for j in range(i + 1, n):
        val_i = (DATA_BA4.iloc[i] - min_ht) / (max_ht - min_ht)
        val_j = (DATA_BA4.iloc[j] - min_ht) / (max_ht - min_ht)
        if not np.isnan(val_i) and not np.isnan(val_j):
            distance = abs(val_i - val_j)
            arr_wik[i, j] = distance
            arr_wik[j, i] = distance

arr_wik_lower = np.tril(arr_wik)

print(np.round(arr_wik_lower, 2))

# %%
from numpy import float64

DATA_BA5 = DATA_AN1['SleepHours']
n = len(DATA_BA5)
arr_sh = np.zeros((n, n), dtype=float64)

min_ht = DATA_BA5.min()
max_ht = DATA_BA5.max()

for i in range(n):
    for j in range(i + 1, n):
        val_i = (DATA_BA5.iloc[i] - min_ht) / (max_ht - min_ht)
        val_j = (DATA_BA5.iloc[j] - min_ht) / (max_ht - min_ht)
        if not np.isnan(val_i) and not np.isnan(val_j):
            distance = abs(val_i - val_j)
            arr_sh[i, j] = distance
            arr_sh[j, i] = distance

arr_sh_lower = np.tril(arr_sh)

print(np.round(arr_sh_lower, 2))

# %%
array_mat = float64(arr_sex + arr_ss + arr_ba_lower + arr_gh_lower + arr_htm_lower + arr_wik_lower + arr_sh_lower) / 7
print(np.round(array_mat, 2))

# %%
def fill_nan(col_name):
    col_idx = DATA_AN1.columns.get_loc(col_name)
    col_dtype = DATA_AN1.dtypes[col_name]
    for i in range(len(DATA_AN1)):
        if pd.isna(DATA_AN1.iloc[i, col_idx]):
            sorted_indices = np.argsort(array_mat[i, :])
            for j in sorted_indices:
                if j != i and not pd.isna(DATA_AN1.iloc[j, col_idx]):
                    j_val = DATA_AN1.iloc[j, col_idx]
                    if col_dtype == bool or str(col_dtype) == "bool":
                        if isinstance(j_val, str):
                            if j_val.lower() == "yes":
                                DATA_AN1.iat[i, col_idx] = True
                            elif j_val.lower() == "no":
                                DATA_AN1.iat[i, col_idx] = False
                            else:
                                DATA_AN1.iat[i, col_idx] = pd.NA
                        else:
                            DATA_AN1.iat[i, col_idx] = bool(j_val)
                    else:
                        DATA_AN1.iat[i, col_idx] = j_val
                    break
fill_nan('SmokerStatus')
fill_nan('HadHeartAttack')
fill_nan('HadAngina')
fill_nan('AlcoholDrinkers')
fill_nan('PhysicalActivities')
fill_nan('GeneralHealth')

DATA_AN1


# %%
arr_cos = np.array(DATA_AN1[['Sex', 'SmokerStatus', 'HadHeartAttack', 'HadAngina', 'AlcoholDrinkers', 'PhysicalActivities', 'HeightInMeters', 'WeightInKilograms', 'SleepHours', 'GeneralHealth']])

dict_cos = {}
from math import sqrt
from numpy.linalg import norm

for i in range(len(arr_cos)):
    for j in range(i + 1, len(arr_cos)):
        if norm(arr_cos[i]) != 0 and norm(arr_cos[j]) != 0:
            cos_sim = sum(arr_cos[i] * arr_cos[j])
            x_ = sqrt(sum(arr_cos[i] * arr_cos[i]))
            y_ = sqrt(sum(arr_cos[j] * arr_cos[j]))
        else:
            cos_sim = np.nan  
        dict_cos[(i, j)] = float(cos_sim) / (x_ * y_)


n = len(DATA_AN1)
arr_cos_lower = np.zeros((n, n), dtype=float)

for (i, j), value in dict_cos.items():
    arr_cos_lower[j, i] = value  
print(arr_cos_lower)

# %%
def num_cal(col_name, df):
    df_name = df[col_name]
    n = len(df_name)
    arr_htm = np.zeros((n, n), dtype=float64)

    min_ht = df_name.min()
    max_ht = df_name.max()
    
    if min_ht == max_ht:
        return np.zeros((n, n), dtype=float64)

    for i in range(n):
        for j in range(i + 1, n):
            val_i = (df_name.iloc[i] - min_ht) / (max_ht - min_ht)
            val_j = (df_name.iloc[j] - min_ht) / (max_ht - min_ht)
            if not np.isnan(val_i) and not np.isnan(val_j):
                distance = abs(val_i - val_j)
                arr_htm[i, j] = distance
                arr_htm[j, i] = distance
    arr_var = np.tril(arr_htm)
    return arr_var

def noun_cal(col_name, df):
    arr1 = np.zeros((len(df), len(df)), dtype=int64)
    for i in range(len(df)):
        for j in range(i):
            a1 = df.iloc[i][col_name]
            a2 = df.iloc[j][col_name]
            arr1[i, j] = 1 if a1 != a2 else 0
    return arr1

def tt_cal(col_name, df, dict):
    df_gh = df[col_name]
    n = len(df_gh)
    arr_gh = np.zeros((n, n), dtype=float64)
    for i in range(n):
        for j in range(i+1, n):
            val_i = dict.get(df_gh.iloc[i], np.nan)
            val_j = dict.get(df_gh.iloc[j], np.nan)
            if not np.isnan(val_i) and not np.isnan(val_j):
                distance = abs(val_i - val_j)
                arr_gh[i, j] = distance
                arr_gh[j, i] = distance  
    n = arr_gh.shape[0]
    arr_gh_lower = np.zeros_like(arr_gh)
    for i in range(n):
        for j in range(i + 1):  
            arr_gh_lower[i, j] = arr_gh[i, j]
    return arr_gh_lower

def boolean_cal_lower_triangle(list_col, df):
    df_bool = df[list_col].copy()
    n = len(df_bool)
    result_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            q = r = s = t = 0
            for col in list_col[1:]:  # Bỏ PersonID
                val_i = df_bool.iloc[i][col]
                val_j = df_bool.iloc[j][col]
                if val_i == 1 and val_j == 1:
                    q += 1
                elif val_i == 1 and val_j == 0:
                    r += 1
                elif val_i == 0 and val_j == 1:
                    s += 1
                else:
                    t += 1
            denominator = q + r + s
            similarity = (r + s) / denominator if denominator != 0 else 0
            result_matrix[i, j] = similarity

    return np.round(result_matrix, 2)

def fill_nan(df, arr_res, col_name):
    col_idx = df.columns.get_loc(col_name)
    col_dtype = df.dtypes[col_name]
    for i in range(len(df)):
        if pd.isna(df.iloc[i, col_idx]):
            sorted_indices = np.argsort(arr_res[i, :])
            for j in sorted_indices:
                if j != i and not pd.isna(df.iloc[j, col_idx]):
                    j_val = df.iloc[j, col_idx]
                    if col_dtype == bool or str(col_dtype) == "bool":
                        if isinstance(j_val, str):
                            if j_val.lower() == "yes":
                                df.iat[i, col_idx] = True
                            elif j_val.lower() == "no":
                                df.iat[i, col_idx] = False
                            else:
                                df.iat[i, col_idx] = pd.NA
                        else:
                            df.iat[i, col_idx] = bool(j_val)
                    else:
                        df.iat[i, col_idx] = j_val
                    break

DATA1 = DATA1.drop('SmokerStatus_Label', axis=1)
DATA_TT = DATA1[:500]
null_cols = DATA_TT.columns[DATA_TT.isnull().any()]
df = {}

for col in null_cols:
    null_rows = DATA_TT[DATA_TT[col].isnull()]
    for idx, row in null_rows.iterrows():
        other_rows = DATA_TT.drop(idx)
        sample_4 = other_rows.sample(n=4)
        df_small = pd.concat([row.to_frame().T, sample_4], ignore_index=True)

        arr_state = noun_cal('State', df_small)
        arr_sex = noun_cal('Sex', df_small)
        dict_gh = {1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1}
        arr_gh = tt_cal('GeneralHealth', df_small, dict_gh)
        arr_phd = num_cal('PhysicalHealthDays', df_small)
        arr_mhd = num_cal('MentalHealthDays', df_small)
        arr_lct = noun_cal('LastCheckupTime', df_small)
        lst = ['PersonID', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos']
        arr_bool = boolean_cal_lower_triangle(lst, df_small)
        arr_sh = num_cal('SleepHours', df_small)
        arr_rt = noun_cal('RemovedTeeth', df_small)
        arr_ss = noun_cal('SmokerStatus', df_small)
        arr_ecu = noun_cal('ECigaretteUsage', df_small)
        arr_rec = noun_cal('RaceEthnicityCategory', df_small)
        dict_ac = {i + 1: round(i / 12, 4) for i in range(13)}
        arr_ac = tt_cal('AgeCategory', df_small, dict_ac)
        arr_him = num_cal('HeightInMeters', df_small)
        arr_wik = num_cal('WeightInKilograms', df_small)
        arr_bmi = num_cal('BMI', df_small)
        arr_tl10 = noun_cal('TetanusLast10Tdap', df_small)

        array_main = float64(arr_state + arr_sex + arr_gh + arr_phd + arr_mhd + arr_lct + arr_bool + arr_sh + arr_rt + arr_ss + arr_ecu + arr_rec + arr_ac + arr_him + arr_wik + arr_bmi + arr_tl10) / 40

        fill_nan(df_small, array_main, col)

        DATA_TT.loc[idx, col] = df_small.loc[0, col]  

DATA_TT

# %%
import pandas as pd

D_test = pd.read_csv(r"E:\Data Science Project\Source\Data\D_DATA.csv")
D_test = D_test[:30]
min_sup = 2
D_temp = D_test.drop('PersonID', axis=1)
lst_col = D_temp.columns
dict_D = {}

for i in range(len(lst_col)):
    for j in range(i + 1, len(lst_col)):
        col1 = lst_col[i]
        col2 = lst_col[j]
        D_pair = D_temp[[col1, col2]].dropna().drop_duplicates()

        for _, row in D_pair.iterrows():
            val1 = row[col1]
            val2 = row[col2]

            count = len(D_temp[(D_temp[col1] == val1) & (D_temp[col2] == val2)])
            if count >= min_sup:
                dict_D[(col1, val1, col2, val2)] = count

print(dict_D)


# %%
import pandas as pd
from itertools import combinations

D_test = pd.read_csv(r"E:\Data Science Project\Source\Data\D_DATA.csv")
num_col = 30
D_test = D_test[:num_col]
min_sup = 2
D_temp = D_test.drop('PersonID', axis=1)
D_temp = D_temp[['Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina']]

def generate_level2(D_temp, min_sup):
    frequent = {}
    cols = list(D_temp.columns)
    for col1, col2 in combinations(cols, 2):
        D_pair = D_temp[[col1, col2]].dropna().drop_duplicates()
        for _, row in D_pair.iterrows():
            val1, val2 = row[col1], row[col2]
            count = len(D_temp[(D_temp[col1] == val1) & (D_temp[col2] == val2)])
            if count >= min_sup:
                key = ((col1, val1), (col2, val2))
                frequent[key] = count
    return frequent

def generate_next_level(prev_dict, D_temp, min_sup):
    next_dict = {}
    keys = list(prev_dict.keys())
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            itemset1, itemset2 = keys[i], keys[j]
            new_items = sorted(set(itemset1 + itemset2), key=lambda x: x[0])
            if len(new_items) != len(set(x[0] for x in new_items)):
                continue  
            if len(new_items) == len(itemset1) + 1:  
                cols = [col for col, val in new_items]
                df_candidate = D_temp[cols].dropna().drop_duplicates()
                for _, row in df_candidate.iterrows():
                    match = True
                    conditions = []
                    for col, val in new_items:
                        conditions.append(D_temp[col] == row[col])
                    count = len(D_temp[conditions[0]])
                    for cond in conditions[1:]:
                        count = len(D_temp[conditions[0] & cond])
                    if count >= min_sup:
                        key = tuple((col, row[col]) for col in cols)
                        next_dict[key] = count
    return next_dict

frequent_dicts = []
level = 2
current_dict = generate_level2(D_temp, min_sup)
frequent_dicts.append(current_dict)

while current_dict:
    print(f"Level {level}: {len(current_dict)} frequent itemsets")
    print(f"Level {level}: {current_dict}")
    current_dict = generate_next_level(current_dict, D_temp, min_sup)
    if current_dict:
        frequent_dicts.append(current_dict)
    level += 1


# %%
cols_to_remove = [
    'State', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
    'LastCheckupTime', 'SleepHours', 'RemovedTeeth', 'SmokerStatus',
    'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory',
    'HeightInMeters', 'WeightInKilograms', 'BMI', 'TetanusLast10Tdap'
]

D_FPG = D_test.drop(columns=cols_to_remove)


# %%
dict_FPG = {}
lst_col = D_FPG.columns
dict_count = {col: (D_FPG[col] == 1).sum() for col in lst_col[1:]}
dict_sorted = dict(sorted(dict_count.items(), key=lambda item: item[1], reverse=True))

for i in range(len(D_FPG)):
    key = D_FPG.iloc[i, 0]
    lst_true = []
    for j in lst_col[1:]:
        if D_FPG.loc[i, j] == 1:
            lst_true.append(j)
    dict_FPG[key] = lst_true  
    
dict_new = {}
for k, v in dict_FPG.items():
    filtered = {col: dict_sorted[col] for col in v if col in dict_sorted}
    dict_new[k] = filtered
for k, v in dict_new.items():
    dict_new[k] = dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
print(dict_new)
keys_to_remove = [k for k, v in dict_sorted.items() if v < min_sup]
for k in keys_to_remove:
    dict_sorted.pop(k)
print(dict_sorted)

person_to_remove = []

for person_id, activity_dict in dict_new.items():
    if any(activity in keys_to_remove for activity in activity_dict):
        person_to_remove.append(person_id)
       
dict_sorted = dict(sorted(dict_sorted.items(), key=lambda item: item[1], reverse=False))
print(dict_sorted)

lst_CPB = []
for k in dict_sorted.keys():
    lst_temp2 = []
    for k1, v1 in dict_new.items():
        if k in v1:
            lst_temp1 = []
            for key, val in v1.items():
                lst_temp1.append(key)
            lst_temp2.append(lst_temp1)
    lst_CPB.append(lst_temp2)
print(lst_CPB)

dictionary_FPG = {}

for key, val in dict_sorted.items():
    for i in lst_CPB:
        mydict_ = {}
        for j in i:
            if key in j:
                for k in j:
                    mydict_[k] = mydict_.get(k, 0) + 1

        lst_tem = [k for k, v in mydict_.items() if v == val]

        if lst_tem:
            dictionary_FPG[tuple(sorted(lst_tem))] = val
            break  

print(dictionary_FPG)


# %%
def count_all_col(D_test, list_columns):
    condition = (D_test[list_columns] == 1).all(axis=1)
    count = condition.sum()
    return float(count) / len(D_test)

converted_dict = {}

for d in frequent_dicts:
    for key_tuple, value in d.items():
        new_key = tuple(k for k, _ in key_tuple)
        converted_dict[new_key] = value

dict_lift2 = {}
for k, v in dictionary_FPG.items():
    dict_temp1 = {}
    dict_temp2 = {}
    dict_temp3 = {}
    dict_total = {}

    if len(k) > 1:
        lst_col1 = list(k[:len(k)//2])  
        lst_col2 = list(k[len(k)//2:])  

        count1 = count_all_col(D_test, lst_col1)
        count2 = count_all_col(D_test, lst_col2)
        count_both = float(v) / len(D_test)

        dict_total[" & ".join(lst_col1)] = count1
        dict_total[" & ".join(lst_col2)] = count2
        dict_total[" & ".join(k)] = count_both

        dict_lift2[" & ".join(k)] = count_both / (count1 * count2)
    else:
        dict_lift2[k[0]] = float(v) / len(D_test)

print(dict_lift2)

dict_lift1 = {}
for k, v in converted_dict.items():
    dict_temp1 = {}
    dict_temp2 = {}
    dict_temp3 = {}
    dict_total = {}

    if len(k) > 1:
        lst_col1 = list(k[:len(k)//2])  
        lst_col2 = list(k[len(k)//2:]) 

        count1 = count_all_col(D_test, lst_col1)
        count2 = count_all_col(D_test, lst_col2)
        count_both = float(v) / len(D_test)

        dict_total[" & ".join(lst_col1)] = count1
        dict_total[" & ".join(lst_col2)] = count2
        dict_total[" & ".join(k)] = count_both

        if count1 > 0 and count2 > 0:
            dict_lift1[" & ".join(k)] = count_both / (count1 * count2)
        else:
            dict_lift1[" & ".join(k)] = 0  
    else:
        dict_lift1[k[0]] = float(v) / len(D_test)

print(dict_lift1)




# %%
relevant_columns = set()
for itemset in dictionary_FPG:
    relevant_columns.update(itemset)

D_test_filtered = D_test[list(relevant_columns)].copy()

def chi_squared_itemset(D_filtered, itemset):
    a = 0  
    total = len(D_filtered)

    for i in range(total):
        row = D_filtered.iloc[i]
        if all(row.get(item, 0) == 1 for item in itemset):
            a += 1

    b = total - a

    p_items = [D_filtered[item].mean() for item in itemset]
    p_expected = 1
    for p in p_items:
        p_expected *= p

    expected_a = total * p_expected
    expected_b = total - expected_a

    if expected_a == 0 or expected_b == 0:
        return 0.0

    chi2 = ((a - expected_a)**2 / expected_a) + ((b - expected_b)**2 / expected_b)
    return round(chi2, 4)


results1 = {}

for itemset in dictionary_FPG:
    chi2_val = chi_squared_itemset(D_test_filtered, itemset)
    results1[itemset] = chi2_val

for itemset, chi2 in results1.items():
    print(f"Itemset {itemset}: χ² = {chi2}")


# %%
results2 = {}

for itemset in converted_dict:
    chi2_vall = chi_squared_itemset(D_test, itemset)
    results2[itemset] = chi2_vall

# In kết quả
for itemset, chi2 in results2.items():
    if chi2 >= 0:
        print(f"Itemset {itemset}: χ² = {chi2}")


