import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def extract_hexcodes(image_path):
    # Load the image
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection and landmark extraction
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define regions of interest (ROI) for hair, skin, and lips
            hair_points = [156, 105, 110, 120]  # Example: hairline area
            skin_points = [31, 32, 33, 34, 35, 36, 100, 110, 120, 130, 150, 160, 148, 149, 152]  # Example: cheek and forehead areas
            lips_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 42, 43, 44, 80, 82, 81, 83, 84, 146, 106]  # Example: lips area

            # Extract average colors from each region
            hair_color = extract_average_color(rgb_img, face_landmarks, hair_points)
            skin_color = extract_average_color(rgb_img, face_landmarks, skin_points)
            lips_color = extract_average_color(rgb_img, face_landmarks, lips_points)

            # Print hex codes
            print("Hex codes -")
            print("Hair color:", hair_color)
            print("Skin color:", skin_color)
            print("Lips color:", lips_color)

            # Visualize the bounding boxes for each region
            visualize_bounding_box(img, face_landmarks, hair_points, (255, 0, 0), "Hair")  # Red for hair
            visualize_bounding_box(img, face_landmarks, skin_points, (0, 255, 0), "Skin")  # Green for skin
            visualize_bounding_box(img, face_landmarks, lips_points, (0, 0, 255), "Lips")  # Blue for lips

    else:
        print("No face detected")

    # Display the image with bounding boxes
    plt.imshow(rgb_img)
    plt.title("Image with Bounding Boxes")
    plt.axis('off')  # Turn off axis
    plt.show()

def extract_average_color(img, landmarks, points):
    # Extract RGB values from image around specified points
    pixels = []
    for idx in points:
        x = int(landmarks.landmark[idx].x * img.shape[1])
        y = int(landmarks.landmark[idx].y * img.shape[0])
        for i in range(-5, 6):  # Expand region by 5 pixels in each direction
            for j in range(-5, 6):
                if 0 <= y + i < img.shape[0] and 0 <= x + j < img.shape[1]:
                    pixels.append(img[y + i, x + j])  # OpenCV uses (y, x) for pixel access

    # Calculate average color
    if len(pixels) == 0:
        return np.array([0, 0, 0])  # Return black if no pixels are selected
    average_color = np.mean(pixels, axis=0)
    average_color = np.uint8(average_color)  # Convert to uint8 format

    return average_color

def visualize_bounding_box(img, landmarks, points, color, label):
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

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Replace 'path_to_your_image.jpg' with your actual image path
image_path = 'D:\pvt\Myntra\MVP\image1.png'
extract_hexcodes(image_path)
