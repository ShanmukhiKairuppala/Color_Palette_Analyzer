from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import re
import cv2
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

color_palettes = {
    'Soft Summer': {
        'Hair': [(0xa0522d, 0xbcafa7), (0xd1b48c, 0xe6ccb2), (0xc0c0c0, 0xc0c0c0)],
        'Lips': [(0xffb6c1, 0xe6e6fa), (0x4b0082, 0x8a2be2)],
        'Skin': [(0xc0c0c0, 0xe0e0e0)],
        'Colors': ['Rose', 'Blue', 'Lavender', 'Sea Green', 'Beige', 'Taupe', 'Grey', 'Pink', 'Blue']
    },
    'Deep Winter': {
        'Hair': [(0x321414, 0x4b0082), (0x808080, 0xc0c0c0), (0x000080, 0x000080)],
        'Lips': [(0x800000, 0x4b0082), (0x8a2be2, 0x800080)],
        'Skin': [(0x321414, 0x808080)],
        'Colors': ['Black', 'Red', 'Green', 'Navy Blue', 'Purple', 'Magenta', 'Charcoal', 'White', 'Burgundy']
    },
    'Light Spring': {
        'Hair': [(0xe6b800, 0xffd700), (0xc68e17, 0xd2b48c), (0xffcc99, 0xffcc99)],
        'Lips': [(0xffc0cb, 0xff69b4), (0xff7f50, 0xff4500)],
        'Skin': [(0xffdab9, 0xffefd5)],
        'Colors': ['Coral', 'Peach', 'Turquoise Blue', 'Pink', 'Yellow', 'Sea Green', 'Cream', 'Blue', 'Orange']
    },
    'Clear Spring': {
        'Hair': [(0xffd700, 0xf0e68c), (0xd2b48c, 0xcd853f), (0xff6347, 0xff7f50)],
        'Lips': [(0xff6347, 0xff4500), (0xff69b4, 0xff1493)],
        'Skin': [(0xffefd5, 0xffdab9)],
        'Colors': ['Orange', 'Pink', 'Yellow', 'Green', 'Turquoise Blue', 'Blue', 'Coral', 'Red', 'Navy Blue']
    },
    'Soft Autumn': {
        'Hair': [(0x8b4513, 0x8b0000), (0xa52a2a, 0xcd5c5c), (0xd2b48c, 0xdeb887)],
        'Lips': [(0xd2691e, 0x8b4513), (0xa52a2a, 0x800000)],
        'Skin': [(0xdeb887, 0xd2b48c)],
        'Colors': ['Olive', 'Gold', 'Rust', 'Grey', 'Rose', 'Brown', 'Rust', 'Olive', 'Taupe']
    },
    'Deep Autumn': {
        'Hair': [(0x2f4f4f, 0x4b0082), (0x800000, 0xa52a2a), (0xcd5c5c, 0x8b0000)],
        'Lips': [(0x800000, 0x4b0082), (0x8a2be2, 0x800080)],
        'Skin': [(0x2f4f4f, 0xcd5c5c)],
        'Colors': ['Olive', 'Orange', 'Teal', 'Brown', 'Rust', 'Green', 'Mustard', 'Burgundy', 'Bronze']
    },
    'Light Summer': {
        'Hair': [(0xe0ffff, 0xb0c4de), (0xb0c4de, 0xc0c0c0), (0xc0c0c0, 0xc0c0c0)],
        'Lips': [(0xffb6c1, 0xe6e6fa), (0x4b0082, 0x8a2be2)],
        'Skin': [(0xe0ffff, 0xc0c0c0)],
        'Colors': ['Blue', 'Rose', 'Lavender', 'Turquoise Blue', 'Pink', 'Grey', 'Mauve', 'Blue', 'Beige']
    },
    'Clear Winter': {
        'Hair': [(0x000000, 0x000000), (0xe0ffff, 0xc0c0c0), (0x808080, 0xc0c0c0)],
        'Lips': [(0xff0000, 0xff1493), (0x800080, 0x4b0082)],
        'Skin': [(0x808080, 0xc0c0c0)],
        'Colors': ['White', 'Black', 'Red', 'Blue', 'Pink', 'Navy Blue', 'Green', 'Grey', 'Purple']
    }
}

csv_file = r'D:\pvt\MVP\Project\Filtered_Myntra_Fasion_Clothing.csv' 
data = pd.read_csv(csv_file)
target_category = 'Indian Wear'
subcategories = ['kurta-sets', 'kurtas', 'sarees', 'dresses']


def recommendations(best_palette):
    palette_name = best_palette
    if palette_name in color_palettes:
        colors = color_palettes[palette_name]['Colors']
        colors = [color.lower() for color in colors] 
    else:
        print(f"Palette '{palette_name}' not found.")
        colors = []  
    
    def contains_color(description, colors):
        pattern = r'\b(?:' + '|'.join(map(re.escape, colors)) + r')\b'
        return re.search(pattern, description.lower()) is not None

    filtered_data = data[
        (data['Category'].str.contains(target_category, case=False)) &
        (data['Individual_category'].str.contains('|'.join(subcategories), case=False)) &
        (data['Description'].apply(lambda x: contains_color(x, colors)))
    ]

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

    return html_content

def extract_hexcodes(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 480))  # Resize for faster processing
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            hair_points = [284, 251, 389, 356]
            cheek_points = [31, 32, 33]
            lips_points = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 287]

            hair_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, hair_points)
            cheek_color = extract_average_color_from_bbox(rgb_img, face_landmarks, cheek_points)
            lips_color = extract_average_color_from_bbox(rgb_img, face_landmarks, lips_points)

            hair_color_rgb = hex_to_rgb(hair_color)
            lips_color_rgb = hex_to_rgb(lips_color)
            cheek_color_rgb = hex_to_rgb(cheek_color)

            matching_scores = []

            for palette_name, palette_ranges in color_palettes.items():
                hair_score = sum(euclidean_distance(hair_color_rgb, hex_to_rgb(hex_code_range_to_mid(hex_range))) for hex_range in palette_ranges['Hair']) / len(palette_ranges['Hair'])
                lips_score = sum(euclidean_distance(lips_color_rgb, hex_to_rgb(hex_code_range_to_mid(hex_range))) for hex_range in palette_ranges['Lips']) / len(palette_ranges['Lips'])
                cheek_score = sum(euclidean_distance(cheek_color_rgb, hex_to_rgb(hex_code_range_to_mid(hex_range))) for hex_range in palette_ranges['Skin']) / len(palette_ranges['Skin'])

                total_score = (hair_score + lips_score + cheek_score) / 3
                matching_scores.append((palette_name, total_score))

            best_palette = min(matching_scores, key=lambda x: x[1])[0]
            
            # print(f"Best matching color palette: {best_palette}")

            # recommendations_html = recommendations(best_palette)

            # # Visualize the bounding boxes for each region
            # visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), f"Hair: {hair_color}")  # Red for hair
            # visualize_bounding_box(rgb_img, face_landmarks, cheek_points, (0, 255, 0), f"Cheek: {cheek_color}")  # Green for cheek
            # visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), f"Lips: {lips_color}")  # Blue for lips
            # bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # # Display the image with bounding boxes using matplotlib
            # plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            # plt.title("Image with Bounding Boxes")
            # plt.axis('off')  # Turn off axis
            # plt.show()
            hex_codes = [hair_color, cheek_color, lips_color]  # Collect hex codes
            best_palette_name = best_palette  # Store best palette name

            recommendations_html = recommendations(best_palette_name)

            result_string = f"Hex codes: {', '.join(hex_codes)}<br>Color Palette: {best_palette_name}<br>{recommendations_html}"

            return result_string


    else:
        print("No face detected")
        return "<p>No face detected in the image.</p>"


def extract_darkest_color_from_bbox(img, landmarks, points):
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)
    roi = img[y_min:y_max, x_min:x_max]
    darkest_color = roi.reshape(-1, 3)[np.argmin(np.sum(roi.reshape(-1, 3), axis=1))]
    hex_color = rgb_to_hex(darkest_color)
    return hex_color

def extract_average_color_from_bbox(img, landmarks, points):
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)
    roi = img[y_min:y_max, x_min:x_max]
    average_color = np.mean(roi.reshape(-1, 3), axis=0)
    hex_color = rgb_to_hex(average_color.astype(int))
    return hex_color

def get_bbox_coordinates(landmarks, points, img):
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')

    for idx in points:
        x = int(landmarks.landmark[idx].x * img.shape[1])
        y = int(landmarks.landmark[idx].y * img.shape[0])
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    return x_min, y_min, x_max, y_max

def visualize_bounding_box(img, landmarks, points, color, label):
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, img)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def euclidean_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))


def hex_code_range_to_mid(hex_range):
    return tuple((hex_to_rgb(hex_range[0])[i] + hex_to_rgb(hex_range[1])[i]) // 2 for i in range(3))


@app.route('/')
def home():
    return render_template('Home.html')  # HTML file for image upload

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']
    if image_file.filename == '':
        return "No selected file", 400
    
    image_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(image_path)
    recommendations_html = extract_hexcodes(image_path)

    if recommendations_html:
        session['recommendations_html'] = recommendations_html
    else:
        session['recommendations_html'] = "No recommendations found."

    return redirect(url_for('result'))

@app.route('/result')
def result():
    recommendations_html = session.get('recommendations_html', "No results available.")
    return render_template('result.html', output=recommendations_html)

if __name__ == '__main__':
    app.run(debug=True)
