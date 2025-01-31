import math
import warnings
from collections import Counter

from skimage.measure import moments_central, moments_normalized, moments_hu
import numpy as np
from skimage import color, filters
from skimage.morphology import disk, closing
from skimage.filters import threshold_otsu, threshold_li
import cv2


class KNN:
    def __init__(self, k, train_filename, predict_image):
        self.k = k
        self.predict_image = predict_image

        # Load train features from CSV file
        self.train_features = np.loadtxt(train_filename, delimiter=',', skiprows=1)  # Skip header
        self.X = self.train_features[:, :-1]
        self.y = self.train_features[:, -1]
        self.category_mapping = {0: 'arandela', 1: 'clavo', 2: 'tornillo', 3: 'tuerca'}

        #print(self.X)
        #print(self.X.shape)
        # Estos valores fueron obtenidos del Standard scaler después de
        # aplicarlo a los training instances (NO MODIFICAR)
        self.features_mean_ = np.array([2.75615255e-03, 1.15365080e-05, 9.56543584e-10, 5.99532635e-10,
                                        1.61668235e-18, 3.00711614e-12, -8.31607848e-21, 5.52256299e-01,
                                        6.69292849e-01,  5.51438738e-01,  1.81247478e-02])
        self.features_var_ = np.array([4.86951262e-06, 2.83091516e-10, 2.33225184e-18, 9.52989144e-19,
                                       1.29119335e-35, 3.28152488e-23, 8.75636961e-39, 1.36462925e-01,
                                       1.10929737e-01, 1.37032663e-01, 1.02454187e-04])

    @staticmethod
    def preprocess_image(img):
        print(f'Original shape: {img.shape}')
        mean_color = img.mean(axis=(0, 1))
        print(f"Mean color (RGB): {mean_color}")

        # Define a threshold for background detection (adjustable)
        threshold = 85

        # Create a mask where pixels close to mean color are set to black
        mask = np.linalg.norm(img - mean_color, axis=-1) < threshold
        img[mask] = [0, 0, 0]  # Convert background to black

        # Convert to grayscale
        # Reducimos el número de canales de 3 a 1
        gray_img = color.rgb2gray(img)

        if gray_img.shape[1] > gray_img.shape[0]:  # Width > Height
            gray_img = np.rot90(gray_img)  # Rotate 90 degrees to make it portrait

        # Apply a median filter
        img_filtered = filters.rank.median(gray_img, disk(5))

        # Apply Li's threshold
        local_li = threshold_li(img_filtered)
        thresh_image = (img_filtered >= local_li).astype(np.uint8) * 255

        # Invert to obtain black background if needed
        if np.mean(thresh_image) > 127:
            thresh_image = cv2.bitwise_not(thresh_image)
            print(f"Inverting colors...")

        # Closing Filtering
        closed_image = closing(thresh_image, disk(10))

        return gray_img, img_filtered, thresh_image, closed_image

    @staticmethod
    def search_contours(image, min_length=100):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.arcLength(cnt, closed=True) >= min_length]

    @staticmethod
    def crop_object(image, contours):
        all_points = np.vstack(contours)
        cv2_contours = np.array(all_points, dtype=np.int32)

        # Compute convex hull of the main contour
        convex_hull = cv2.convexHull(cv2_contours)

        # Get bounding rectangle around the convex hull
        x, y, w, h = cv2.boundingRect(convex_hull)

        # Crop the image using the bounding rectangle
        cropped = image[y:y + h, x:x + w]

        # Adjust the contours by shifting them to the cropped image coordinate system
        adjusted_contours = [cnt - np.array([x, y]) for cnt in cv2_contours]
        adjusted_hull = convex_hull - np.array([x, y])

        return cropped, adjusted_contours, adjusted_hull

    @staticmethod
    def compute_features(image, contours, hull):
        # Ratio ConvexHullArea / minEnclosingCircleArea
        (x, y), radius = cv2.minEnclosingCircle(hull)
        circle_area = math.pi * (radius ** 2)
        print(f"Circle Center: ({x}, {y}), Radius: {radius}, Area: {circle_area}")

        hull_area = cv2.contourArea(hull)
        print(f"Hull area: {hull_area}")

        area_ratio = hull_area / circle_area
        print(f'Area ratio: {area_ratio}')

        # Axis Length Aspect Ratio
        ellipse = cv2.fitEllipse(hull)
        major_axis = ellipse[1][1]
        minor_axis = ellipse[1][0]
        print(f'Minor axis: {minor_axis} - Major axis: {major_axis}')
        axis_aspect_ratio = minor_axis / major_axis
        if major_axis != 0:  # To avoid division by zero
            eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
        else:
            eccentricity = 0  # If major axis is 0, set eccentricity to 0 (degenerate case)
        print(f'Aspect ratio: {axis_aspect_ratio} - Eccentricity: {eccentricity}')


        # Ratio Perimeter / Area
        perimeter = cv2.arcLength(hull, closed=True)
        per_area_aspect_ratio = perimeter / hull_area
        print(f'Aspect ratio: {per_area_aspect_ratio}')

        # Hu Moments
        mu = moments_central(image)
        nu = moments_normalized(mu)
        hu_moments = moments_hu(nu)

        features = np.append(hu_moments, np.array([area_ratio, axis_aspect_ratio, eccentricity, per_area_aspect_ratio]))
        return features

    def draw_convex_hull(self, cropped_image, adjusted_hull):
        """
        Draw the convex hull contour on the cropped image in red.

        Parameters:
            cropped_image (np.array): The cropped object image.
            adjusted_hull (list): The convex hull contour.

        Returns:
            np.array: Cropped image with convex hull drawn in red.
        """
        # Convert to BGR if the image is grayscale
        if len(cropped_image.shape) == 2:  # Grayscale
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

        # Ensure adjusted_hull is in correct format
        if len(adjusted_hull) > 0:
            hull_array = np.array([adjusted_hull], dtype=np.int32)

            # Draw the convex hull in red (BGR: (0, 0, 255))
            cv2.polylines(cropped_image, hull_array, isClosed=True, color=(255, 0, 0), thickness=4)

        return cropped_image  # Returns the numpy array with the red convex hull

    def launch_knn(self):
        # Preprocess the original image
        gray_img, img_filtered, thresh_image, closed_image = self.preprocess_image(self.predict_image)

        # Get the object contours
        contours = self.search_contours(closed_image, min_length=100)

        # Crop the image around the detected object
        cropped_image, adjusted_contours, adjusted_hull = self.crop_object(closed_image, contours)
        hull_image = self.draw_convex_hull(cropped_image, adjusted_hull)
        # Compute the object set of features
        predict_features = self.compute_features(cropped_image,adjusted_contours, adjusted_hull)

        # Scale the features
        predict_features = predict_features.reshape(-1, 1)  # Ensure it's (11,1)
        predict_features_scaled = (predict_features - self.features_mean_.reshape(-1, 1)) / np.sqrt(self.features_var_.reshape(-1, 1))

        # Drop Hu_Moments_7
        predict_features_scaled = np.delete(predict_features_scaled, 6)

        # KNN Algorithm
        if self.X.shape[1] != predict_features_scaled.shape[0]:
            raise f"Invalid input shape. It should be {self.X.shape} but it is {predict_features_scaled.shape}."
        if self.k <= len(np.unique(self.y)):
            warnings.warn('K is set to a value less than total voting groups.')

        distances = []
        for index, instance in enumerate(self.X):
            euclidean_distance = np.linalg.norm(instance - predict_features_scaled)
            distances.append([euclidean_distance, self.y[index]])

        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result = Counter(votes).most_common(1)[0]

        vote_result_categorical = self.category_mapping[int(vote_result[0])]  # Gives the name of the class ('arandela' for example)
        confidence = vote_result[1] / self.k

        return vote_result_categorical, confidence, hull_image, np.delete(predict_features, 6)
