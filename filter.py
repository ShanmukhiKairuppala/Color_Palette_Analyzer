import pandas as pd

# Load the dataset
csv_file = r'D:\pvt\Myntra\MVP\Myntra Fasion Clothing.csv'  # Update with the path to your CSV file
data = pd.read_csv(csv_file)

# Define the target categories and subcategories
target_category_indian_wear = 'Indian Wear'
subcategories_indian_wear = ['sarees', 'kurtas', 'kurta-sets', 'dresses']

target_category_western_wear = 'Western'
subcategories_western_wear = ['dresses']

# Filter the data
filtered_data = data[
    (data['category_by_Gender'].str.lower() == 'women') &
    (
        ((data['Category'].str.lower() == target_category_indian_wear.lower()) & (data['Individual_category'].str.lower().isin(subcategories_indian_wear))) |
        ((data['Category'].str.lower() == target_category_western_wear.lower()) & (data['Individual_category'].str.lower().isin(subcategories_western_wear)))
    )
]

# Display the filtered data
print(filtered_data)

# Save the filtered data to a new CSV file if needed
filtered_csv_file = r'D:\pvt\Myntra\MVP\Filtered_Myntra_Fasion_Clothing.csv'
filtered_data.to_csv(filtered_csv_file, index=False)
print(f"Filtered data saved to {filtered_csv_file}")
