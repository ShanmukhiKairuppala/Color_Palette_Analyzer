# Displays product details in terminal based on user input for color

import pandas as pd

# Load the dataset
csv_file = r'D:\pvt\Myntra\MVP\Myntra Fasion Clothing.csv'  # Update with the path to your CSV file
data = pd.read_csv(csv_file)

# Define the target category and subcategories
target_category = 'Indian Wear'
subcategories = ['kurta-sets', 'kurtas', 'sarees', 'dresses']

# Ask user for the color
color = input("Enter the color you want to filter by: ").strip().lower()

# Initialize results dictionary
results = {subcategory: [] for subcategory in subcategories}

# Function to find products matching the criteria
def find_products(data, category, subcategory, color):
    matching_products = []
    for _, row in data.iterrows():
        if (category.lower() in row['Category'].lower() and 
            subcategory.lower() in row['Individual_category'].lower() and 
            color in row['Description'].lower()):
            matching_products.append(row)
    return matching_products

# Find products for each subcategory
for subcategory in subcategories:
    results[subcategory] = find_products(data, target_category, subcategory, color)

# Display the results
for subcategory, products in results.items():
    if products:
        print(f"\nSubcategory: {subcategory.title()}")
        for product in products:
            description = product['Description']
            color_info = color if color in description.lower() else 'Not specified'
            
            print(f"\nProduct URL: {product['URL']}")
            print(f"Product ID: {product['Product_id']}")
            print(f"Brand: {product['BrandName']}")
            print(f"Discount Price: Rs. {product['DiscountPrice (in Rs)']}")
            print(f"Original Price: Rs. {product['OriginalPrice (in Rs)']}")
            print(f"Discount Offer: {product['DiscountOffer']}")
            print(f"Size Options: {product['SizeOption']}")
            print(f"Ratings: {product['Ratings']}")
            print(f"Reviews: {product['Reviews']}")
            print(f"Description: {description}")
            print(f"Color: {color_info}")
            print("-" * 40)
    else:
        print(f"No products found for subcategory: {subcategory.title()} with color {color}")
        print("-" * 40)
