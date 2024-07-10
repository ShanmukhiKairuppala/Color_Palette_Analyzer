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
            #159, 158, 157, 173, 133, 155, 154      65,55,53,62,74
            hair_points = [156,105,110,120]  # Example: hairline area
            skin_points = [31,32,33,34,35,36,100,110,120,130,150,160,148,149,152]  # Example: cheek and forehead areas
            lips_points = [11,12,13,14,15,16,17,18,19,20,38,39 ,40,41,42,43,44,80,82,81,83,84,146,106]   # Example: lips area
            # Visualize the bounding boxes for each region
            visualize_bounding_box(img, face_landmarks, hair_points, (255, 0, 0), "Hair")  # Red for hair
            visualize_bounding_box(img, face_landmarks, skin_points, (0, 255, 0), "Skin")  # Green for skin
            visualize_bounding_box(img, face_landmarks, lips_points, (0, 0, 255), "Lips")  # Blue for lips

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

    # Display the image with bounding boxes
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Image with Bounding Boxes")
    plt.axis('off')  # Turn off axis
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        return "#000000"  # Return black if no pixels are selected
    average_color = np.mean(pixels, axis=0)
    average_color = np.uint8(average_color)  # Convert to uint8 format

    # Convert average color to hex format
    hex_color = rgb_to_hex(average_color)

    return hex_color

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

def rgb_to_hex(rgb):
    # Convert RGB to hex format
    return '#{:02x}{:02x}{:02x}'.format(rgb[2], rgb[1], rgb[0])  # Convert BGR to RGB

# Replace 'path_to_your_image.jpg' with your actual image path
image_path = 'D:\pvt\Myntra\MVP\image1.png'
extract_hexcodes(image_path)

# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt

# Initialize mediapipe FaceMesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# def extract_hexcodes(image_path):
#     # Load the image
#     img = cv2.imread(image_path)
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Perform face detection and landmark extraction
#     results = face_mesh.process(rgb_img)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # Define regions of interest (ROI) for different facial features
#             left_cheek_points = [4, 5, 6, 7, 8, 9, 10, 11, 206, 205, 204]
#             right_cheek_points = [12, 13, 14, 15, 16, 17, 18, 19, 234, 233, 232]
#             upper_lip_points = [61, 62, 63, 64, 65, 66, 67, 68, 296, 407, 318]
#             lower_lip_points = [61, 146, 91, 181, 84, 17, 314, 405, 312, 311, 310]
#             forehead_points = [10, 11, 12, 13, 14, 15, 16, 23, 24, 25]
#             hairline_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#             above_hairline_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#             ears_points = [234, 454, 348, 232, 142, 366, 374, 380, 385]
#             eyes_points = [33, 133, 246, 7, 55, 175, 139, 20, 230]
#             nose_points = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#             # Visualize the bounding boxes for each region (example)
#             visualize_bounding_box(img, face_landmarks, left_cheek_points, (255, 0, 0), "Left Cheek")
#             visualize_bounding_box(img, face_landmarks, right_cheek_points, (0, 255, 0), "Right Cheek")
#             visualize_bounding_box(img, face_landmarks, upper_lip_points, (0, 0, 255), "Upper Lip")
#             visualize_bounding_box(img, face_landmarks, lower_lip_points, (255, 255, 0), "Lower Lip")
#             visualize_bounding_box(img, face_landmarks, forehead_points, (255, 0, 255), "Forehead")
#             visualize_bounding_box(img, face_landmarks, hairline_points, (0, 255, 255), "Hairline")
#             visualize_bounding_box(img, face_landmarks, above_hairline_points, (128, 128, 128), "Above Hairline")
#             visualize_bounding_box(img, face_landmarks, ears_points, (128, 0, 128), "Ears")
#             visualize_bounding_box(img, face_landmarks, eyes_points, (0, 128, 128), "Eyes")
#             visualize_bounding_box(img, face_landmarks, nose_points, (128, 128, 0), "Nose")

#             # Extract average colors from each region (example)
#             left_cheek_color = extract_average_color(rgb_img, face_landmarks, left_cheek_points)
#             right_cheek_color = extract_average_color(rgb_img, face_landmarks, right_cheek_points)
#             upper_lip_color = extract_average_color(rgb_img, face_landmarks, upper_lip_points)
#             lower_lip_color = extract_average_color(rgb_img, face_landmarks, lower_lip_points)
#             forehead_color = extract_average_color(rgb_img, face_landmarks, forehead_points)
#             hairline_color = extract_average_color(rgb_img, face_landmarks, hairline_points)
#             above_hairline_color = extract_average_color(rgb_img, face_landmarks, above_hairline_points)
#             ears_color = extract_average_color(rgb_img, face_landmarks, ears_points)
#             eyes_color = extract_average_color(rgb_img, face_landmarks, eyes_points)
#             nose_color = extract_average_color(rgb_img, face_landmarks, nose_points)

#             # Print hex codes (example)
#             print("Hex codes -")
#             print("Left Cheek color:", left_cheek_color)
#             print("Right Cheek color:", right_cheek_color)
#             print("Upper Lip color:", upper_lip_color)
#             print("Lower Lip color:", lower_lip_color)
#             print("Forehead color:", forehead_color)
#             print("Hairline color:", hairline_color)
#             print("Above Hairline color:", above_hairline_color)
#             print("Ears color:", ears_color)
#             print("Eyes color:", eyes_color)
#             print("Nose color:", nose_color)

#         # Display the image with bounding boxes using matplotlib
#         plt.imshow(img)
#         plt.title("Image with Bounding Boxes")
#         plt.axis('off')  # Turn off axis
#         plt.show()

#     else:
#         print("No face detected")

# def visualize_bounding_box(img, landmarks, points, color, label):
#     x_min, x_max, y_min, y_max = get_bounding_box(landmarks, points, img.shape[1], img.shape[0])

#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# def get_bounding_box(landmarks, points, img_width, img_height):
#     # Initialize min and max coordinates
#     x_min, x_max = img_width, 0
#     y_min, y_max = img_height, 0

#     # Update min and max coordinates based on landmark points
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img_width)
#         y = int(landmarks.landmark[idx].y * img_height)
#         if x < x_min:
#             x_min = x
#         if x > x_max:
#             x_max = x
#         if y < y_min:
#             y_min = y
#         if y > y_max:
#             y_max = y

#     # Add some margin to the bounding box
#     margin = 10
#     x_min = max(0, x_min - margin)
#     y_min = max(0, y_min - margin)
#     x_max = min(img_width, x_max + margin)
#     y_max = min(img_height, y_max + margin)

#     return x_min, x_max, y_min, y_max

# def extract_average_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(img[y, x])  # OpenCV uses (y, x) for pixel access

#     # Calculate average color
#     average_color = np.mean(pixels, axis=0)
#     average_color = np.uint8(average_color)  # Convert to uint8 format

#     # Convert average color to hex format
#     hex_color = rgb_to_hex(average_color)

#     return hex_color

# def rgb_to_hex(rgb):
#     # Convert RGB to hex format
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# # Replace 'path_to_your_image.jpg' with your actual image path
# image_path = 'D:\pvt\Myntra\MVP\image1.png'
# extract_hexcodes(image_path)
