
import pandas as pd
import re

# Load the dataset
csv_file = r'D:\pvt\Myntra\MVP\Filtered_Myntra_Fasion_Clothing.csv' 
data = pd.read_csv(csv_file)

target_category = 'Indian Wear'
subcategories = ['kurta-sets', 'kurtas', 'sarees', 'dresses']

# Ask user for the color
color = input("Enter the color you want to filter by: ").strip().lower()

# Function to check if the color is a standalone word in the description
def contains_color(description, color):
    pattern = rf'\b{re.escape(color)}\b'
    return re.search(pattern, description.lower()) is not None

# Filter the dataset based on category, subcategory, and color
filtered_data = data[
    (data['Category'].str.contains(target_category, case=False)) &
    (data['Individual_category'].str.contains('|'.join(subcategories), case=False)) &
    (data['Description'].apply(lambda x: contains_color(x, color)))
]

# Generate the HTML page
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Filtered Products</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .product-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .product { flex: 1 1 calc(33.333% - 20px); box-sizing: border-box; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .product img { width: 100%; height: auto; }
        .product-details { margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Filtered Products</h1>
    <div class="product-container">
"""

for _, product in filtered_data.iterrows():
    html_content += f"""
    <div class="product">
        <img src="{product['URL']}" alt="{product['Description']}">
        <div class="product-details">
            <p><strong>Product ID:</strong> {product['Product_id']}</p>
            <p><strong>Brand:</strong> {product['BrandName']}</p>
            <p><strong>Category:</strong> {product['Category']}</p>
            <p><strong>Subcategory:</strong> {product['Individual_category']}</p>
            <p><strong>Description:</strong> {product['Description']}</p>
            <p><strong>Discount Price:</strong> Rs. {product['DiscountPrice (in Rs)']}</p>
            <p><strong>Original Price:</strong> Rs. {product['OriginalPrice (in Rs)']}</p>
            <p><strong>Discount Offer:</strong> {product['DiscountOffer']}</p>
            <p><strong>Size Options:</strong> {product['SizeOption']}</p>
            <p><strong>Ratings:</strong> {product['Ratings']}</p>
            <p><strong>Reviews:</strong> {product['Reviews']}</p>
            <p><a href="{product['URL']}">Buy Now</a></p>
        </div>
    </div>
    """

html_content += """
    </div>
</body>
</html>
"""

# Save the HTML content to a file
html_file_path = r'D:\pvt\Myntra\MVP\filtered_products.html'  
with open(html_file_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML file has been generated and saved to {html_file_path}")

