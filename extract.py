import cv2
import numpy as np
import mediapipe as mp

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
            hair_points = [10, 338, 297, 332, 284, 251, 389, 356]  # Example: hairline area
            skin_points = [234, 454, 234, 10, 152, 148, 176, 148, 151]  # Example: cheek and forehead areas
            lips_points = list(range(61, 78))    # Example: lips area

            # Extract average colors from each region
            hair_color = extract_average_color(rgb_img, face_landmarks, hair_points)
            skin_color = extract_average_color(rgb_img, face_landmarks, skin_points)
            lips_color = extract_average_color(rgb_img, face_landmarks, lips_points)

            # Print hex codes
            print("Hex codes -")
            print("Hair color:", hair_color)
            print("Skin color:", skin_color)
            print("Lips color:", lips_color)
    else:
        print("No face detected")

def extract_average_color(img, landmarks, points):
    # Extract RGB values from image around specified points
    pixels = []
    for idx in points:
        x = int(landmarks.landmark[idx].x * img.shape[1])
        y = int(landmarks.landmark[idx].y * img.shape[0])
        pixels.append(img[y, x])  # OpenCV uses (y, x) for pixel access

    # Calculate average color
    average_color = np.mean(pixels, axis=0)
    average_color = np.uint8(average_color)  # Convert to uint8 format

    # Convert average color to hex format
    hex_color = rgb_to_hex(average_color)

    return hex_color

def rgb_to_hex(rgb):
    # Convert RGB to hex format
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Replace 'path_to_your_image.jpg' with your actual image path
image_path = 'D:\pvt\Myntra\MVP\image4.jpg'
extract_hexcodes(image_path)
