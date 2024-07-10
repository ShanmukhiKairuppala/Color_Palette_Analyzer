# import cv2
# import numpy as np
# import mediapipe as mp

# # Initialize mediapipe FaceMesh
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
#             # Define regions of interest (ROI) for hair, skin, and lips
#             # hair_points = [10, 338, 297, 332, 284, 251, 389, 356]  # Example: hairline area
#             # skin_points = [234, 454, 234, 10, 152, 148, 176, 148, 151]  # Example: cheek and forehead areas
#             # lips_points = list(range(61, 78))    # Example: lips area
#             hair_points = [156,105,110,120]  # Example: hairline area
#             skin_points = [31,32,33,34,35,36,100,110,120,130,150,160,148,149,152]  # Example: cheek and forehead areas
#             lips_points = [11,12,13,14,15,16,17,18,19,20,38,39 ,40,41,42,43,44,80,82,81,83,84,146,106]   # Example: lips area
           
#             # Extract average colors from each region
#             hair_color = extract_average_color(rgb_img, face_landmarks, hair_points)
#             skin_color = extract_average_color(rgb_img, face_landmarks, skin_points)
#             lips_color = extract_average_color(rgb_img, face_landmarks, lips_points)

#             # Print hex codes
#             print("Hex codes -")
#             print("Hair color:", hair_color)
#             print("Skin color:", skin_color)
#             print("Lips color:", lips_color)
#     else:
#         print("No face detected")

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

#--------------------------------------------------

# import cv2
# import numpy as np
# import mediapipe as mp

# # Initialize mediapipe FaceMesh
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
#             # Define regions of interest (ROI) for hair, skin, and lips
#             hair_points = [156, 105, 110, 120]  # Example: hairline area
#             skin_points = [31, 32, 33, 34, 35, 36, 100, 110, 120, 130, 150, 160, 148, 149, 152]  # Example: cheek and forehead areas
#             lips_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 42, 43, 44, 80, 82, 81, 83, 84, 146, 106]  # Example: lips area

#             # Extract colors from each region based on the specified criteria
#             hair_color = extract_darkest_color(rgb_img, face_landmarks, hair_points)
#             skin_color = extract_most_repeated_color(rgb_img, face_landmarks, skin_points)
#             lips_color = extract_maximum_color(rgb_img, face_landmarks, lips_points)

#             # Print hex codes
#             print("Hex codes -")
#             print("Hair color:", hair_color)
#             print("Skin color:", skin_color)
#             print("Lips color:", lips_color)
#     else:
#         print("No face detected")

# def extract_darkest_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(img[y, x])  # OpenCV uses (y, x) for pixel access

#     # Find the darkest color (minimum sum of RGB values)
#     darkest_color = min(pixels, key=lambda color: np.sum(color))
    
#     # Convert the darkest color to hex format
#     hex_color = rgb_to_hex(darkest_color)
#     return hex_color

# def extract_most_repeated_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(tuple(img[y, x]))  # OpenCV uses (y, x) for pixel access

#     # Find the most repeated color
#     most_repeated_color = max(set(pixels), key=pixels.count)
    
#     # Convert the most repeated color to hex format
#     hex_color = rgb_to_hex(np.array(most_repeated_color))
#     return hex_color

# def extract_maximum_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(img[y, x])  # OpenCV uses (y, x) for pixel access

#     # Find the maximum color (maximum sum of RGB values)
#     max_color = max(pixels, key=lambda color: np.sum(color))
    
#     # Convert the maximum color to hex format
#     hex_color = rgb_to_hex(max_color)
#     return hex_color

# def rgb_to_hex(rgb):
#     # Convert RGB to hex format
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# # Replace 'path_to_your_image.jpg' with your actual image path
# image_path = 'D:\pvt\Myntra\MVP\image4.jpg'
# extract_hexcodes(image_path)

#--------------------------------------------------
# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt

# # Initialize mediapipe FaceMesh
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
#             # Define regions of interest (ROI) for hair, skin, and lips
#             hair_points = [156, 105, 110, 120]  # Example: hairline area
#             skin_points = [31, 32, 33, 34, 35, 36, 100, 110, 120, 130, 150, 160, 148, 149, 152]  # Example: cheek and forehead areas
#             lips_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 42, 43, 44, 80, 82, 81, 83, 84, 146, 106]  # Example: lips area

#             # Visualize the bounding boxes for each region
#             visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), "Hair")  # Red for hair
#             visualize_bounding_box(rgb_img, face_landmarks, skin_points, (0, 255, 0), "Skin")  # Green for skin
#             visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), "Lips")  # Blue for lips

#             # Extract colors from each region based on the specified criteria
#             hair_color = extract_darkest_color(rgb_img, face_landmarks, hair_points)
#             skin_color = extract_most_repeated_color(rgb_img, face_landmarks, skin_points)
#             lips_color = extract_maximum_color(rgb_img, face_landmarks, lips_points)

#             # Print hex codes
#             print("Hex codes -")
#             print("Hair color:", hair_color)
#             print("Skin color:", skin_color)
#             print("Lips color:", lips_color)
#     else:
#         print("No face detected")

#     # Display the image with bounding boxes using matplotlib
#     plt.imshow(rgb_img)
#     plt.title("Image with Bounding Boxes")
#     plt.axis('off')  # Turn off axis
#     plt.show()

# def extract_darkest_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(img[y, x])  # OpenCV uses (y, x) for pixel access

#     # Find the darkest color (minimum sum of RGB values)
#     darkest_color = min(pixels, key=lambda color: np.sum(color))
    
#     # Convert the darkest color to hex format
#     hex_color = rgb_to_hex(darkest_color)
#     return hex_color

# def extract_most_repeated_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(tuple(img[y, x]))  # OpenCV uses (y, x) for pixel access

#     # Find the most repeated color
#     most_repeated_color = max(set(pixels), key=pixels.count)
    
#     # Convert the most repeated color to hex format
#     hex_color = rgb_to_hex(np.array(most_repeated_color))
#     return hex_color

# def extract_maximum_color(img, landmarks, points):
#     # Extract RGB values from image around specified points
#     pixels = []
#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         pixels.append(img[y, x])  # OpenCV uses (y, x) for pixel access

#     # Find the maximum color (maximum sum of RGB values)
#     max_color = max(pixels, key=lambda color: np.sum(color))
    
#     # Convert the maximum color to hex format
#     hex_color = rgb_to_hex(max_color)
#     return hex_color

# def visualize_bounding_box(img, landmarks, points, color, label):
#     x_min = y_min = float('inf')
#     x_max = y_max = float('-inf')

#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         if x < x_min:
#             x_min = x
#         if y < y_min:
#             y_min = y
#         if x > x_max:
#             x_max = x
#         if y > y_max:
#             y_max = y

#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# def rgb_to_hex(rgb):
#     # Convert RGB to hex format
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# # Replace 'path_to_your_image.jpg' with your actual image path
# image_path = 'D:\pvt\Myntra\MVP\image5.webp'
# extract_hexcodes(image_path)

#--------------------------------------------------

# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt

# # Initialize mediapipe FaceMesh
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
#             # Define regions of interest (ROI) for hair, skin, and lips
#             hair_points = [156, 105, 110, 120]  # Example: hairline area
#             skin_points = [31, 32, 33, 34, 35, 36, 100, 110, 120, 130, 150, 160, 148, 149, 152]  # Example: cheek and forehead areas
#             lips_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 42, 43, 44, 80, 82, 81, 83, 84, 146, 106]  # Example: lips area

#             # Visualize the bounding boxes for each region
#             visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), "Hair", img)  # Red for hair
#             visualize_bounding_box(rgb_img, face_landmarks, skin_points, (0, 255, 0), "Skin", img)  # Green for skin
#             visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), "Lips", img)  # Blue for lips

#             # Extract colors from each region based on the specified criteria
#             hair_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, hair_points, img)
#             skin_color = extract_most_repeated_color_from_bbox(rgb_img, face_landmarks, skin_points, img)
#             lips_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, lips_points, img)

#             # Print hex codes
#             print("Hex codes -")
#             print("Hair color:", hair_color)
#             print("Skin color:", skin_color)
#             print("Lips color:", lips_color)
#     else:
#         print("No face detected")

#     # Display the image with bounding boxes using matplotlib
#     plt.imshow(rgb_img)
#     plt.title("Image with Bounding Boxes")
#     plt.axis('off')  # Turn off axis
#     plt.show()

# def extract_darkest_color_from_bbox(img, landmarks, points, original_img):
#     # Get bounding box coordinates
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)

#     # Extract the region of interest (ROI) from the image
#     roi = img[y_min:y_max, x_min:x_max]

#     # Find the darkest color (minimum sum of RGB values) in the ROI
#     darkest_color = min(roi.reshape(-1, 3), key=lambda color: np.sum(color))

#     # Convert the darkest color to hex format
#     hex_color = rgb_to_hex(darkest_color)
#     return hex_color

# def extract_most_repeated_color_from_bbox(img, landmarks, points, original_img):
#     # Get bounding box coordinates
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)

#     # Extract the region of interest (ROI) from the image
#     roi = img[y_min:y_max, x_min:x_max]

#     # Reshape the ROI to a list of pixels
#     pixels = [tuple(p) for p in roi.reshape(-1, 3)]

#     # Find the most repeated color in the ROI
#     most_repeated_color = max(set(pixels), key=pixels.count)

#     # Convert the most repeated color to hex format
#     hex_color = rgb_to_hex(np.array(most_repeated_color))
#     return hex_color

# def get_bbox_coordinates(landmarks, points, img):
#     x_min = y_min = float('inf')
#     x_max = y_max = float('-inf')

#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         if x < x_min:
#             x_min = x
#         if y < y_min:
#             y_min = y
#         if x > x_max:
#             x_max = x
#         if y > y_max:
#             y_max = y

#     return x_min, y_min, x_max, y_max

# def visualize_bounding_box(img, landmarks, points, color, label, original_img):
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# def rgb_to_hex(rgb):
#     # Convert RGB to hex format
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# # Replace 'path_to_your_image.jpg' with your actual image path
# image_path = 'D:\pvt\Myntra\MVP\image3.jpg'
# extract_hexcodes(image_path)

#--------------------------------------------------
#taking more time - dark hair pixel from box, most repeated pixel for skin and lips 
# import cv2
# import numpy as np
# import mediapipe as mp
# import matplotlib.pyplot as plt

# # Initialize mediapipe FaceMesh
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
#             # Define regions of interest (ROI) for hair, skin, and lips
#             hair_points = [156, 105, 110, 120]  # Example: hairline area
#             skin_points = [31, 32, 33, 34, 35, 36, 100, 110, 120, 130, 150, 160, 148, 149, 152]  # Example: cheek and forehead areas
#             lips_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 42, 43, 44, 80, 82, 81, 83, 84, 146, 106]  # Example: lips area

#             # Visualize the bounding boxes for each region
#             visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), "Hair", img)  # Red for hair
#             visualize_bounding_box(rgb_img, face_landmarks, skin_points, (0, 255, 0), "Skin", img)  # Green for skin
#             visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), "Lips", img)  # Blue for lips

#             # Extract colors from each region based on the specified criteria
#             hair_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, hair_points, img)
#             skin_color = extract_most_repeated_color_from_bbox(rgb_img, face_landmarks, skin_points, img)
#             lips_color = extract_most_repeated_color_from_bbox(rgb_img, face_landmarks, lips_points, img)

#             # Print hex codes
#             print("Hex codes -")
#             print("Hair color:", hair_color)
#             print("Skin color:", skin_color)
#             print("Lips color:", lips_color)
#     else:
#         print("No face detected")

#     # Display the image with bounding boxes using matplotlib
#     plt.imshow(rgb_img)
#     plt.title("Image with Bounding Boxes")
#     plt.axis('off')  # Turn off axis
#     plt.show()

# def extract_darkest_color_from_bbox(img, landmarks, points, original_img):
#     # Get bounding box coordinates
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)

#     # Extract the region of interest (ROI) from the image
#     roi = img[y_min:y_max, x_min:x_max]

#     # Find the darkest color (minimum sum of RGB values) in the ROI
#     darkest_color = min(roi.reshape(-1, 3), key=lambda color: np.sum(color))

#     # Convert the darkest color to hex format
#     hex_color = rgb_to_hex(darkest_color)
#     return hex_color

# def extract_most_repeated_color_from_bbox(img, landmarks, points, original_img):
#     # Get bounding box coordinates
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)

#     # Extract the region of interest (ROI) from the image
#     roi = img[y_min:y_max, x_min:x_max]

#     # Reshape the ROI to a list of pixels
#     pixels = [tuple(p) for p in roi.reshape(-1, 3)]

#     # Find the most repeated color in the ROI
#     most_repeated_color = max(set(pixels), key=pixels.count)

#     # Convert the most repeated color to hex format
#     hex_color = rgb_to_hex(np.array(most_repeated_color))
#     return hex_color

# def get_bbox_coordinates(landmarks, points, img):
#     x_min = y_min = float('inf')
#     x_max = y_max = float('-inf')

#     for idx in points:
#         x = int(landmarks.landmark[idx].x * img.shape[1])
#         y = int(landmarks.landmark[idx].y * img.shape[0])
#         if x < x_min:
#             x_min = x
#         if y < y_min:
#             y_min = y
#         if x > x_max:
#             x_max = x
#         if y > y_max:
#             y_max = y

#     return x_min, y_min, x_max, y_max

# def visualize_bounding_box(img, landmarks, points, color, label, original_img):
#     x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# def rgb_to_hex(rgb):
#     # Convert RGB to hex format
#     return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# # Replace 'path_to_your_image.jpg' with your actual image path
# image_path = 'D:\pvt\Myntra\MVP\image3.jpg'
# extract_hexcodes(image_path)

#--------------------------------------------------

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
    img = cv2.resize(img, (640, 480))  # Resize for faster processing
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection and landmark extraction
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define regions of interest (ROI) for hair, skin, and lips
            hair_points = [156, 105, 110, 120]  # Example: hairline area
            skin_points = [31, 32, 33, 34, 35, 36, 100, 110, 120, 130, 150, 160, 148, 149, 152]  # Example: cheek and forehead areas
            lips_points = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 42, 43, 44, 80, 82, 81, 83, 84, 146, 106]  # Example: lips area

            # Visualize the bounding boxes for each region
            visualize_bounding_box(rgb_img, face_landmarks, hair_points, (255, 0, 0), "Hair", img)  # Red for hair
            visualize_bounding_box(rgb_img, face_landmarks, skin_points, (0, 255, 0), "Skin", img)  # Green for skin
            visualize_bounding_box(rgb_img, face_landmarks, lips_points, (0, 0, 255), "Lips", img)  # Blue for lips

            # Extract colors from each region based on the specified criteria
            hair_color = extract_darkest_color_from_bbox(rgb_img, face_landmarks, hair_points, img)
            skin_color = extract_most_repeated_color_from_bbox(rgb_img, face_landmarks, skin_points, img)
            lips_color = extract_most_repeated_color_from_bbox(rgb_img, face_landmarks, lips_points, img)

            # Print hex codes
            print("Hex codes -")
            print("Hair color:", hair_color)
            print("Skin color:", skin_color)
            print("Lips color:", lips_color)
    else:
        print("No face detected")

    # Display the image with bounding boxes using matplotlib
    plt.imshow(rgb_img)
    plt.title("Image with Bounding Boxes")
    plt.axis('off')  # Turn off axis
    plt.show()

def extract_darkest_color_from_bbox(img, landmarks, points, original_img):
    # Get bounding box coordinates
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)

    # Extract the region of interest (ROI) from the image
    roi = img[y_min:y_max, x_min:x_max]

    # Find the darkest color (minimum sum of RGB values) in the ROI
    darkest_color = min(roi.reshape(-1, 3), key=lambda color: np.sum(color))

    # Convert the darkest color to hex format
    hex_color = rgb_to_hex(darkest_color)
    return hex_color

def extract_most_repeated_color_from_bbox(img, landmarks, points, original_img):
    # Get bounding box coordinates
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)

    # Extract the region of interest (ROI) from the image
    roi = img[y_min:y_max, x_min:x_max]

    # Reshape the ROI to a list of pixels
    pixels = [tuple(p) for p in roi.reshape(-1, 3)]

    # Find the most repeated color in the ROI
    most_repeated_color = max(set(pixels), key=pixels.count)

    # Convert the most repeated color to hex format
    hex_color = rgb_to_hex(np.array(most_repeated_color))
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

def visualize_bounding_box(img, landmarks, points, color, label, original_img):
    x_min, y_min, x_max, y_max = get_bbox_coordinates(landmarks, points, original_img)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def rgb_to_hex(rgb):
    # Convert RGB to hex format
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Replace 'path_to_your_image.jpg' with your actual image path
image_path = 'D:\pvt\Myntra\MVP\image2.jpg'
extract_hexcodes(image_path)
