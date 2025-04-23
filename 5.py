import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data)
df['species'] = iris.target
# Optional: rename columns manually if needed
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# --- Seaborn plots ---
# 1. Seaborn Scatterplot
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title("Seaborn Scatterplot")
plt.show()
# 2. Seaborn Boxplot
sns.boxplot(data=df, x='species', y='petal_length')
plt.title("Seaborn Boxplot")
plt.show()
# --- Matplotlib plots ---
# 3. Matplotlib Line Plot
plt.plot(df['sepal_length'], label='Sepal Length')
plt.plot(df['petal_length'], label='Petal Length')
plt.title("Matplotlib Line Plot")
plt.show()
# 4. Matplotlib Histogram
plt.hist(df['sepal_width'])
plt.title("Matplotlib Histogram")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.show()