import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("salaries.csv")

print(df.head())
print(df.info())
print(df.describe())

# Salary distribution
sns.histplot(df["salary_in_usd"], bins=50)
plt.title("Salary Distribution")
plt.show()

# Experience vs salary
sns.boxplot(x="experience_level", y="salary_in_usd", data=df)
plt.show() 