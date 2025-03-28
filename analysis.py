import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats




df = pd.read_csv("Team Stats(Sheet1) (2).csv")


df["GoalsF"] = df["GFMod"] / df["Even Strength Time "] * 60
df["GoalsA"] = df["GAMod"] / df["Even Strength Time "] * 60
data = df.drop(columns=["Game","Game ID","Even Strength Time "])
data = data.drop(columns=["XG","NF+","NF-","Odd+","Odd-","XGF","XGA"])
data =data.drop(columns=["Cycle","Rush","Faceoff","Fore"])
data = data.drop(columns=["CE","UE"])
data["Points"]= data["Result"].apply(lambda x: 2 if x == "W" else 1 if x == "T" else 0)
data["Penalty Min Differential"]= data["PPT"] - data["SHT"]
data.drop(columns=["Result"],inplace=True)



# Select the target variable (e.g., GFMod) and compute correlations
#target_variable = "Points"

# 

# Spearman correlation
#spearman_corr = data.corr(method="spearman")[target_variable].sort_values(ascending=False)
#print("🔹 Spearman Correlation 🔹\n", spearman_corr)

corr_matrix = data.corr(method="spearman")


# Function to compute p-values for correlation matrices
def compute_p_values(df_corr, df_data):
    p_values = df_corr.copy()
    for col in df_corr.columns:
        for row in df_corr.index:
            if row != col:
                _, p = stats.spearmanr(df_data[row].dropna(), df_data[col].dropna())
                p_values.at[row, col] = p
            else:
                p_values.at[row, col] = None  # No p-values for self-correlation
    return p_values
# Extract only the "Points" correlations
points_corr = corr_matrix[["Points"]].sort_values(by="Points", ascending=False)




rename_dict = {
    "GF": "Goals For",
    "SA": "Shots Against",
    "SF": "Shots For",
    "CES": "Shots per conttolled Entry",
    "OddF/60": "Odd-Man For/60",
    "Rush/60": "Rush Shots/60",
    "NFA/60": "Net Front Against/60",
    "UES": "Shots per uncontrolled entry",
    "Cycle/60": "Shots off the cycle/60",
    "GA": "Goals Against",
    "CE/60": "Controlled Entries/60",
    "OddA/60": "Odd-Man Against/60",
    "Controlled Entry Rate": "Controlled Entry %",
    "NFF/60": "Net Front For/60",
    "CEX": "Controlled Exits Rate",
    "Faceoff/60": "Shots of Faceoffs/60",
    "Fore/60": "Shots of the Forechecks/60",
    "X": "Successful Zone Exit Rate",
    "UE/60": "Uncontrolled Entries/60",
    "SD": "Shot Differntial",
    "AD": "Grade A Differential",
    "NFD": "Net Front Differential",
    "OddD": "Odd Man Differential",
    "A+": "Scoring Chances For",
    "A-": "Scoring Chances Against",
    "ST Imp": "Special Teams Goal Dif"
}

points_corr.rename(index=rename_dict, inplace=True)
points_corr.drop(index=["GFMod","GAMod","GoalsF","GoalsA"], errors="ignore", inplace=True)



plt.figure(figsize=(6, 10))
ax = sns.heatmap(points_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Waterloo Points correlations")
plt.yticks(fontsize=8)



# Adjust text alignment to shift annotations left
for text in ax.texts:
    text.set_ha("left")
    text.set_position((text.get_position()[0] - 0.4, text.get_position()[1]))


# Matrix for Goals For 
goals_forcorr = corr_matrix[["GoalsF"]].sort_values(by="GoalsF",ascending=False)
goals_forcorr.drop(index=["GF","GAMod","GFMod","GoalsA","SHT","PPT","Penalty Min Differential","Points","SF","SA","ST Imp"],inplace=True)
goals_forcorr.rename(index=rename_dict,inplace=True,errors="ignore")
# Create heatmap and assign the correct 'ax'
plt.figure(figsize=(6, 10))
ax = sns.heatmap(goals_forcorr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Goals For Correlations")
plt.yticks(fontsize=8)

# Adjust text positioning to move annotations to the left
for text in ax.texts:
    text.set_ha("left")  # Align left
    text.set_x(text.get_position()[0] - 0.4)  # Shift more left

plt.show()





goals_againstcorr = corr_matrix[["GoalsA"]].sort_values(by="GoalsA",ascending=False)
goals_againstcorr.drop(index=["GA","GFMod","GoalsF","GAMod","SHT","PPT","Penalty Min Differential","Points","SF","SA","ST Imp"],inplace=True)
goals_againstcorr.rename(index=rename_dict,errors="ignore",inplace=True)
plt.figure(figsize=(6, 10))
ax = sns.heatmap(goals_againstcorr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Goals Against Correlations")
plt.yticks(fontsize=8)

for text in ax.texts:
    text.set_ha("left")
    text.set_position((text.get_position()[0] - 0.4, text.get_position()[1]))

plt.show()



# Not Much here
oddfcorr = corr_matrix[["OddF/60"]].sort_values(by="OddF/60",ascending=False)
oddfcorr.drop(index=["GA","SA","SF"],inplace=True)
selected_indexes = ["X", "CE/60", "CEX", "UE/60", "Controlled Entry Rate"]
oddfcorr = oddfcorr.loc[oddfcorr.index.intersection(selected_indexes)]
oddfcorr.rename(index=rename_dict,errors="ignore",inplace=True)
plt.figure(figsize=(6, 10))
ax = sns.heatmap(oddfcorr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Odd Man Corr")
plt.yticks(fontsize=8)

for text in ax.texts:
    text.set_ha("left")
    text.set_position((text.get_position()[0] - 0.4, text.get_position()[1]))

plt.show()


