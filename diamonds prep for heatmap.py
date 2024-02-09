import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('diamonds2022.csv')

# Print number of unique diamonds
print(df['diamond_id'].nunique())

# Select the required columns
df = df[['diamond_id','size', 'cut', 'color', 'clarity', 'symmetry', 'total_sales_price']]

# Define the mappings
cut_mapping = {'None': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Ideal': 4, 'Excellent': 5}
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_mapping = {'I3': 0, 'I2': 1, 'I1': 2, 'SI3': 3, 'SI2': 4, 'SI1': 5, 'VS2': 6, 'VS1': 7, 'VVS2': 8, 'VVS1': 9, 'IF': 10}
symmetry_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}

# Apply the mappings
df['cut'] = df['cut'].map(cut_mapping)
df['color'] = df['color'].map(color_mapping)
df['clarity'] = df['clarity'].map(clarity_mapping)
df['symmetry'] = df['symmetry'].map(symmetry_mapping)

# Separate the diamond_id and target variable
diamond_id = df['diamond_id']
target = df['total_sales_price']
df = df.drop(columns=['diamond_id', 'total_sales_price'])

# Normalize the features
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Merge back with the diamond_id and target variable
df_normalized = pd.concat([diamond_id, df_normalized, target], axis=1)

# Calculate the correlation matrix
corr = df_normalized.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
print(df_normalized.head())

# Change the format to let us do an heatmap
df_long = pd.melt(df_normalized, id_vars=['diamond_id'], value_vars=['size', 'cut', 'color', 'clarity', 'symmetry', 'total_sales_price'],
                  var_name='Attribute', value_name='Value')

# Print number of rows
print(df_long.shape)

# Print number of unique diamonds
print(df_long['diamond_id'].nunique())

print(df_long.head())
df_long.to_csv('diamonds_long_format.csv', index=False)