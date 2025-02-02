import math
import warnings
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import mode
from skimage import color, filters
from skimage.morphology import disk, closing
from skimage.filters import threshold_otsu, threshold_li
import cv2


class ImageRecognitionAlgorithm:
    # Estos valores fueron obtenidos del Standard scaler después de
    # aplicarlo a los training instances (NO MODIFICAR)
    features_mean_ = np.array([0.38477562, 0.5522563 , 0.55143874])
    features_var_ = np.array([0.0844822 , 0.13646293, 0.13703266])

    category_mapping = {0: 'arandela', 1: 'clavo', 2: 'tornillo', 3: 'tuerca'}

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
        if not contours:
            return None  # Return None if no contours are found
        return max(contours, key=cv2.contourArea)  # Return the longest contour
        #return [cnt for cnt in contours if cv2.arcLength(cnt, closed=True) >= min_length]

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
        # Ensure contours and hull are numpy arrays
        contours = np.array(contours, dtype=np.float32) if not isinstance(contours, np.ndarray) else contours
        hull = np.array(hull, dtype=np.float32) if not isinstance(hull, np.ndarray) else hull

        # Compute Hu_Moments of the Convex Hull
        moments = cv2.moments(hull)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Compute Area and Perimeter
        area = cv2.contourArea(contours)
        perimeter = cv2.arcLength(contours, True)

        # Compute Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        # Compute Convex Hull Features
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, closed=True)

        solidity = area / hull_area if hull_area != 0 else 0
        convexity = hull_perimeter / perimeter if perimeter != 0 else 0

        # Compute Min Enclosing Circle
        (x, y), radius = cv2.minEnclosingCircle(hull)
        circle_area = math.pi * (radius ** 2)

        area_ratio = hull_area / circle_area if circle_area != 0 else 0

        # Compute Ellipse Fitting
        if len(hull) >= 5:  # cv2.fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(hull)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            axis_aspect_ratio = minor_axis / major_axis if major_axis != 0 else 0
            eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2)) if major_axis != 0 else 0
        else:
            major_axis, minor_axis, axis_aspect_ratio, eccentricity = 0, 0, 0, 0

        # Compute Perimeter/Area Ratio
            per_area_aspect_ratio = perimeter / hull_area if hull_area != 0 else 0

        # Solo conservamos el primer momento de Hu, el circleAreaRatio y el eccentricity
        features = np.append(hu_moments[0], np.array([area_ratio, eccentricity]))
        return features

    def scale_features(self, features):
        features = features.reshape(-1)  # Ensure (3,) shape
        mean = self.features_mean_.reshape(-1)  # Ensure (3,) shape
        var = self.features_var_.reshape(-1)  # Ensure (3,) shape

        if features.shape[0] != mean.shape[0]:
            raise ValueError(f"Feature dimension mismatch: {features.shape[0]} vs {mean.shape[0]}")

        predict_features_scaled = (features - mean) / np.sqrt(var)
        return predict_features_scaled  # Ensuring (3,) shape
    @staticmethod
    def draw_convex_hull(cropped_image, adjusted_hull):
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

    def plot_3d_comparison(self, predict_features_scaled, train_features_scaled, train_labels, filename):
        """
        Plot a 3D comparison of the training features and the predicted feature.

        Parameters:
            predict_features_scaled (np.array): Scaled features of the predicted object.
            train_features_scaled (np.array): Scaled features of the training data.
            train_labels (np.array): Labels of the training data.
            filename (str): Name of the file to save the plot.
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(14, 10))  # Increased figure size for better spacing
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=65)

        # Plot the training features
        scatter = ax.scatter(train_features_scaled[:, 0], train_features_scaled[:, 1], train_features_scaled[:, 2],
                             c=train_labels, cmap='viridis', label='Training Data', alpha=0.6)

        # Plot the predicted feature
        ax.scatter(predict_features_scaled[0], predict_features_scaled[1], predict_features_scaled[2],
                   c='red', marker='X', s=200, label='Predicted Instance')

        # Create a custom legend for the training data colors
        unique_labels = np.unique(train_labels)
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
        legend_elements = [Patch(facecolor=cmap(i), label=f'Label {self.category_mapping[int(label)]}') for i, label in enumerate(unique_labels)]

        # Add the predicted feature to the legend
        legend_elements.append(Patch(facecolor='red', label='Predicted Feature'))

        # Add the combined legend to the plot
        ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.15, 1), loc='upper left')

        plt.title('3D Comparison of Training Features and Predicted Instance', pad=20)  # Add padding to the title

        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)  # Manually adjust subplot margins
        # Add labels with increased padding
        ax.set_xlabel('Feature 1 (Scaled Hu Moment 1)', labelpad=15, )  # Increased padding for x-axis
        ax.set_ylabel('Feature 2 (Scaled Circle Area Ratio)', labelpad=15)  # Increased padding for y-axis
        ax.set_zlabel('Feature 3 (Scaled Eccentricity)', labelpad=15)  # Increased padding for z-axis
        # Save the plot
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()


class KNN(ImageRecognitionAlgorithm):
    def __init__(self, k, train_filename, predict_image):
        self.k = k
        self.predict_image = predict_image

        # Load train features from CSV file
        self.train_features = np.loadtxt(train_filename, delimiter=',', skiprows=1)  # Skip header
        self.X = self.train_features[:, :-1]
        self.y = self.train_features[:, -1]

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
        predict_features_scaled =  self.scale_features(predict_features)

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

        self.plot_3d_comparison(predict_features_scaled, self.X, self.y, '3d_plot_comparison.png')

        return vote_result_categorical, confidence, hull_image, predict_features, predict_features_scaled


class KMeans(ImageRecognitionAlgorithm):
    def __init__(self, train_filename, predict_image, k=4, max_iters=100, tol=1e-4, n_init = 10):
        """
        Parameters:
        - X: numpy array of shape (n_samples, n_features), the input data.
        - k: int, the number of clusters.
        - max_iters: int, maximum number of iterations per run.
        - tol: float, tolerance for convergence.
        - n_init: int, number of times to run K-Means with different initializations.
        """
        self.predict_image = predict_image

        # Load train features from CSV file
        self.train_features = np.loadtxt(train_filename, delimiter=',', skiprows=1)  # Skip header
        self.X = self.train_features[:, :-1]
        self.y = self.train_features[:, -1]

        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init


    @staticmethod
    def initialize_centroids(X, k):
        """Randomly chose k centroids from the dataset."""
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    @staticmethod
    def kmeans_plusplus_initialization(X, k):
        """
        K-Means++ initialization for selecting initial centroids.
        """
        n_samples, n_features = X.shape

        # Step 1: Randomly select the first centroid
        centroids = [X[np.random.choice(n_samples)]]

        for _ in range(1, k):
            # Step 2: Compute squared distances to the nearest centroid
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])

            # Step 3: Select the next centroid with probability proportional to distances
            probabilities = distances / distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_centroid_idx])

        return np.array(centroids)

    @staticmethod
    def assign_clusters(X, centroids):
        """Assign each point to the nearest centroid."""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    @staticmethod
    def update_centroids(X, labels, k):
        """Compute new centroids as the mean of points in each cluster."""
        return np.array([X[labels == i].mean(axis=0) for i in range(k)])

    def predict(self, predict, centroids):
        return self.assign_clusters(predict, centroids)

    @staticmethod
    def cluster_to_label_mapping(y_true, y_pred):
        """Finds the most frequent true label for each cluster."""
        unique_clusters = np.unique(y_pred)
        cluster_mapping = {}

        for cluster in unique_clusters:
            mask = y_pred == cluster
            most_common_label = mode(y_true[mask]).mode
            cluster_mapping[int(cluster)] = int(most_common_label)

        return cluster_mapping

    @staticmethod
    def clustering_accuracy(y_true, y_pred, cluster_mapping):
        """Computes the clustering accuracy based on cluster-label mapping."""
        correct_count = np.sum([cluster_mapping[label] == y_true[i] for i, label in enumerate(y_pred)])
        total_count = len(y_true)
        return correct_count / total_count

    def plot_3d_comparison(self, predict_features_scaled, train_features_scaled, train_labels, filename):
        """
        Plot a 3D comparison of the training features and the predicted feature.

        Parameters:
            predict_features_scaled (np.array): Scaled features of the predicted object.
            train_features_scaled (np.array): Scaled features of the training data.
            train_labels (np.array): Labels of the training data.
            filename (str): Name of the file to save the plot.
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(14, 10))  # Increased figure size for better spacing
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=65)

        # Plot the training features
        scatter = ax.scatter(train_features_scaled[:, 0], train_features_scaled[:, 1], train_features_scaled[:, 2],
                             c=train_labels, cmap='viridis', label='Training Data', alpha=0.6)

        # Plot the predicted feature
        ax.scatter(predict_features_scaled[0], predict_features_scaled[1], predict_features_scaled[2],
                   c='red', marker='X', s=200, label='Predicted Instance')

        # Create a custom legend for the training data colors
        unique_labels = np.unique(train_labels)
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
        legend_elements = [Patch(facecolor=cmap(i), label=f'Group {int(label)}') for i, label in enumerate(unique_labels)]

        # Add the predicted feature to the legend
        legend_elements.append(Patch(facecolor='red', label='Predicted Feature'))

        # Add the combined legend to the plot
        ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.15, 1), loc='upper left')

        plt.title('3D Comparison of Training Features and Predicted Instance', pad=20)  # Add padding to the title

        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)  # Manually adjust subplot margins
        # Add labels with increased padding
        ax.set_xlabel('Feature 1 (Scaled Hu Moment 1)', labelpad=15, )  # Increased padding for x-axis
        ax.set_ylabel('Feature 2 (Scaled Circle Area Ratio)', labelpad=15)  # Increased padding for y-axis
        ax.set_zlabel('Feature 3 (Scaled Eccentricity)', labelpad=15)  # Increased padding for z-axis
        # Save the plot
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()

    def launch_kmeans(self):
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
        predict_features_scaled =  self.scale_features(predict_features)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = self.kmeans_plusplus_initialization(self.X, self.k)

            for _ in range(self.max_iters):
                labels = self.assign_clusters(self.X, centroids)
                new_centroids = self.update_centroids(self.X, labels, self.k)

                # Check for convergence
                if np.linalg.norm(new_centroids - centroids) < self.tol:
                    break

                centroids = new_centroids

            # Compute inertia (sum of squared distances)
            inertia = np.sum([np.linalg.norm(self.X[labels == j] - centroids[j]) ** 2 for j in range(self.k)])

            # Update best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        print(f"Resulting centroids: {best_centroids}")

        # Cluster to Label mapping
        mapping = self.cluster_to_label_mapping(np.array(self.y), np.array(best_labels))
        print("Cluster to Label Mapping:", mapping)

        accuracy = self.clustering_accuracy(self.y, best_labels, mapping)
        print(f"Clustering Accuracy: {accuracy:.2%}")

        # Predict image
        predicted_group = self.predict(predict_features_scaled.reshape(1, -1), best_centroids)[0]
        predicted_label_int = mapping[predicted_group]
        predicted_label = self.category_mapping[predicted_label_int]
        print(f"Predicted group: {predicted_group} - Mapped Label: {predicted_label_int} - Label: {predicted_label}")

        self.plot_3d_comparison(predict_features_scaled, self.X, best_labels, '3d_plot_comparison.png')

        return predicted_group, predicted_label, accuracy, hull_image, predict_features, predict_features_scaled