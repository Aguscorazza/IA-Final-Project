import math
import warnings
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import mode
from skimage import color, filters
from skimage.morphology import disk, binary_closing
from skimage.filters import threshold_otsu, threshold_li
import cv2


class ImageRecognitionAlgorithm:
    # Estos valores fueron obtenidos del Standard scaler después de
    # aplicarlo a los training instances (NO MODIFICAR)
    features_mean_ = np.array([ 3.84775619e-01,  2.05266479e-01,  6.06069527e-03,  4.88774242e-03,
        8.02606812e-05,  3.82024021e-03, -1.51912884e-08,  8.57088350e+04,
        2.21141749e+03,  4.75608075e-01,  6.76078372e-01,  8.21467071e-01,
        8.22579423e-01,  5.52256299e-01,  6.69292849e-01,  5.51438738e-01,
        2.21659637e-02])

    features_var_ = np.array([8.44821990e-02, 9.30134667e-02, 6.30604977e-05, 5.11569028e-05,
       2.15418556e-08, 4.01467314e-05, 1.20044068e-14, 3.19930778e+09,
       3.91801831e+06, 1.28554142e-01, 9.72304043e-02, 3.23354363e-02,
       3.26561282e-02, 1.36462925e-01, 1.10929737e-01, 1.37032663e-01,
       1.17376805e-04])

    features_names = ['Hu1_hull', 'Hu2_hull', 'Hu3_hull', 'Hu4_hull', 'Hu5_hull', 'Hu6_hull',
    'Hu7_hull', 'Area', 'Perimeter', 'Circularity', 'Hull_Circularity',
    'Solidity', 'Convexity', 'Circle_Area_Ratio', 'Axis_Aspect_Ratio',
    'Eccentricity', 'Perimeter_Area_Ratio']

    chosen_features = ['Hu1_hull', 'Hull_Circularity', 'Circle_Area_Ratio', 'Eccentricity']

    category_mapping = {0: 'arandela', 1: 'clavo', 2: 'tornillo', 3: 'tuerca'}

    @staticmethod
    def preprocess_image(img):
        print(f'Original shape: {img.shape}')
        mean_color = img.mean(axis=(0, 1))
        print(f"Mean color (RGB): {mean_color}")

        # Threshold para la detección del fondo
        threshold = 85

        # Máscara donde los pixeles cerca del color del fondo se vuelven negros
        mask = np.linalg.norm(img - mean_color, axis=-1) < threshold
        img[mask] = [0, 0, 0]  # Convert background to black

        # Convertir a escala de grises
        # Reducimos el número de canales de 3 a 1
        gray_img = color.rgb2gray(img)

        if gray_img.shape[1] > gray_img.shape[0]:
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
        closed_image = binary_closing(thresh_image, disk(10))

        return gray_img, img_filtered, thresh_image, closed_image

    @staticmethod
    def search_contours(image, min_length=100):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # None si ningún contorno
        return max(contours, key=cv2.contourArea)  # Devuelve el contorno mas grande

        #return [cnt for cnt in contours if cv2.arcLength(cnt, closed=True) >= min_length]

    @staticmethod
    def crop_object(image, contours):
        all_points = np.vstack(contours)
        cv2_contours = np.array(all_points, dtype=np.int32)

        # Calcula el convex_hull de los contornos de cada imagen
        convex_hull = cv2.convexHull(cv2_contours)

        # Get bounding rectangle around the convex hull
        x, y, w, h = cv2.boundingRect(convex_hull)

        # Crop the image using the bounding rectangle
        cropped = image[y:y + h, x:x + w]

        # Ajuste de coordenadas de los contornos y convex_hull
        # debido al recorte de la imagen
        adjusted_contours = [cnt - np.array([x, y]) for cnt in cv2_contours]
        adjusted_hull = convex_hull - np.array([x, y])

        return cropped, adjusted_contours, adjusted_hull

    @staticmethod
    def compute_features(image, contours, hull):
        # Nos aseguramos que los contornos y el convex hull son np.array
        contours = np.array(contours, dtype=np.float32) if not isinstance(contours, np.ndarray) else contours
        hull = np.array(hull, dtype=np.float32) if not isinstance(hull, np.ndarray) else hull

        # Cálculo de los momentos de Hu (convex_hull)
        moments = cv2.moments(hull)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Cálculo del área y perímetro de los contornos de la imagen
        area = cv2.contourArea(contours)
        perimeter = cv2.arcLength(contours, True)

        # Circularidad
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        # Caracteristicas geometricas del convex hull
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, closed=True)

        solidity = area / hull_area if hull_area != 0 else 0
        convexity = hull_perimeter / perimeter if perimeter != 0 else 0
        hull_circularity = (4 * np.pi * hull_area) / (hull_perimeter ** 2) if hull_perimeter != 0 else 0

        # Cálculo del menor circulo que encierra al convex hull
        (x, y), radius = cv2.minEnclosingCircle(hull)
        circle_area = math.pi * (radius ** 2)

        area_ratio = hull_area / circle_area if circle_area != 0 else 0

        # Ellipse fitting
        if len(hull) >= 5:  # cv2.fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(hull)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            axis_aspect_ratio = minor_axis / major_axis if major_axis != 0 else 0
            eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2)) if major_axis != 0 else 0
        else:
            major_axis, minor_axis, axis_aspect_ratio, eccentricity = 0, 0, 0, 0

        # Perimetro / Area (No muy util)
        per_area_aspect_ratio = perimeter / hull_area if hull_area != 0 else 0


        features = np.concatenate((hu_moments, [area, perimeter, circularity, hull_circularity,
                                    solidity, convexity,area_ratio, axis_aspect_ratio,
                                    eccentricity, per_area_aspect_ratio]))
        return features

    def scale_features(self, features):
        features = features.reshape(-1)  # Ensure (n,) shape
        mean = self.features_mean_.reshape(-1)  # Ensure (n,) shape
        var = self.features_var_.reshape(-1)  # Ensure (n,) shape

        if features.shape[0] != mean.shape[0]:
            raise ValueError(f"Feature dimension mismatch: {features.shape[0]} vs {mean.shape[0]}")

        predict_features_scaled = (features - mean) / np.sqrt(var)
        return predict_features_scaled  # Ensuring (n,) shape

    def select_features(self, features):
        # Get indices of chosen features
        indices = [self.features_names.index(feature) for feature in self.chosen_features]
        return features[indices]

    @staticmethod
    def draw_convex_hull(cropped_image, adjusted_hull):
        # Convert to BGR if the image is grayscale
        if len(cropped_image.shape) == 2:  # Grayscale
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

        # Si existe el convex_hull, lo añadimos a la imagen
        if len(adjusted_hull) > 0:
            hull_array = np.array([adjusted_hull], dtype=np.int32)
            cv2.polylines(cropped_image, hull_array, isClosed=True, color=(255, 0, 0), thickness=4)

        return cropped_image  # Returns the numpy array with the red convex hull

    def plot_3d_comparison(self, predict_features_scaled, train_features_scaled, train_labels, filename):

        # Create a 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=65)

        # Plot the training instances
        scatter = ax.scatter(train_features_scaled[:, 0], train_features_scaled[:, 1], train_features_scaled[:, 2],
                             c=train_labels, cmap='viridis', label='Datos de entrenamiento', alpha=0.6)

        # Plot the predicted instance
        ax.scatter(predict_features_scaled[0], predict_features_scaled[1], predict_features_scaled[2],
                   c='red', marker='X', s=200, label='Punto de datos de entrada')

        # Create a custom legend for the training data colors
        unique_labels = np.unique(train_labels)
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
        legend_elements = [Patch(facecolor=cmap(i), label=f'Clase {self.category_mapping[int(label)]}') for i, label in enumerate(unique_labels)]

        # Add the predicted feature to the legend
        legend_elements.append(Patch(facecolor='red', label='Punto de datos de entrada'))
        ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.15, 1), loc='upper left')

        plt.title('Gráfico 3D de comparación', pad=20)

        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)
        ax.set_xlabel(f'Característica 1 ({self.chosen_features[0]})', labelpad=15, )
        ax.set_ylabel(f'Característica 2 ({self.chosen_features[1]})', labelpad=15)
        ax.set_zlabel(f'Característica 3 ({self.chosen_features[2]})', labelpad=15)
        # Save the plot
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()


class KNN(ImageRecognitionAlgorithm):
    def __init__(self, k, train_filename, predict_image):
        self.k = k
        self.predict_image = predict_image

        # Carga las características de entrenamiento a partir de un archivo CSV.
        self.train_features = np.loadtxt(train_filename, delimiter=',', skiprows=1)  # Skip header
        self.X = self.train_features[:, :-1]
        self.y = self.train_features[:, -1]

        # Seleccionamos las caracteristicas
        self.indices = [self.features_names.index(feature) for feature in self.chosen_features]
        self.X = self.X[:, self.indices]

    def launch_knn(self):
        # Procesa la imagen origianl
        gray_img, img_filtered, thresh_image, closed_image = self.preprocess_image(self.predict_image)

        # Nos aseguramos que la imagen esté en el formato correcto (uint8)
        if closed_image.dtype == np.bool_:
            closed_image = closed_image.astype(np.uint8) * 255

        # Búsqueda de contornos
        contours = self.search_contours(closed_image, min_length=100)

        # Recorta la imagen alrededor del contorno principal (convex_hull)
        cropped_image, adjusted_contours, adjusted_hull = self.crop_object(closed_image, contours)

        # Construye la imagen que luego se muestra en la interfaz de usuario
        hull_image = self.draw_convex_hull(cropped_image, adjusted_hull)

        # Cálculo de caracteristicas
        predict_features = self.compute_features(cropped_image,adjusted_contours, adjusted_hull)

        # Procesamiento de caracteristicas (cambio de escala)
        predict_features_scaled =  self.scale_features(predict_features)

        # Seleccion de caracteristicas
        predict_features = self.select_features(predict_features)
        predict_features_scaled = self.select_features(predict_features_scaled)

        # Algoritmo KNN)
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

        vote_result_categorical = self.category_mapping[int(vote_result[0])]  # Devuelve el nombre de la clase ('arandela' por ejemplo)
        confidence = vote_result[1] / self.k

        if len(predict_features_scaled) == 3:
            self.plot_3d_comparison(predict_features_scaled, self.X, self.y, '3d_plot_comparison.png')

        return vote_result_categorical, confidence, hull_image, predict_features, predict_features_scaled, self.chosen_features


class KMeans(ImageRecognitionAlgorithm):
    def __init__(self, train_filename, predict_image, k=4, max_iters=100, tol=1e-4, n_init = 10):
        self.predict_image = predict_image

        # Carga los datos de entrenamiento a partir de un archivo CSV
        self.train_features = np.loadtxt(train_filename, delimiter=',', skiprows=1)  # Skip header
        self.X = self.train_features[:, :-1]
        self.y = self.train_features[:, -1]

        # Seleccionamos las caracteristicas
        self.indices = [self.features_names.index(feature) for feature in self.chosen_features]
        self.X = self.X[:, self.indices]

        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init


    @staticmethod
    def initialize_centroids(X, k):
        """Elige k centroides aleatoriamente."""
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    @staticmethod
    def kmeans_plusplus_initialization(X, k):
        """
        Inicialización Kmeans++ para seleccionar los k centroides iniciales.
        """
        n_samples, n_features = X.shape

        # Paso 1 : Elegir aleatoriamente el primer centroide
        centroids = [X[np.random.choice(n_samples)]]

        for _ in range(1, k):
            # Paso 2: Calcular la distancia al centroide más cercano
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])

            # Paso 3: Selecciona el siguiente centroide con probabilidad proporcional a la distancia
            probabilities = distances / distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_centroid_idx])

        return np.array(centroids)

    @staticmethod
    def assign_clusters(X, centroids):
        """Asigna cada punto al centroide más cercano."""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    @staticmethod
    def update_centroids(X, labels, k):
        """Calcula los nuevos centroides como la media de los puntos en cada cluster."""
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
        """Calcula la precision del clustering en base al mapeo."""
        correct_count = np.sum([cluster_mapping[label] == y_true[i] for i, label in enumerate(y_pred)])
        total_count = len(y_true)
        return correct_count / total_count

    def plot_3d_comparison(self, predict_features_scaled, train_features_scaled, train_labels, filename):

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=35, azim=65)

        # Datos de entrenamiento
        scatter = ax.scatter(train_features_scaled[:, 0], train_features_scaled[:, 1], train_features_scaled[:, 2],
                             c=train_labels, cmap='viridis', label='Datos de entrenamiento', alpha=0.6)

        # Dato de entrada
        ax.scatter(predict_features_scaled[0], predict_features_scaled[1], predict_features_scaled[2],
                   c='red', marker='X', s=200, label='Punto de datos de entrada')

        unique_labels = np.unique(train_labels)
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
        legend_elements = [Patch(facecolor=cmap(i), label=f'Grupo {int(label)}') for i, label in enumerate(unique_labels)]

        legend_elements.append(Patch(facecolor='red', label='Punto de datos de entrada'))

        ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.15, 1), loc='upper left')

        plt.title('Gráfico 3D de comparación', pad=20)

        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)
        ax.set_xlabel('Característica 1 (Scaled Hu Moment 1)', labelpad=15)
        ax.set_ylabel('Característica 2 (Scaled Ratio de área circular)', labelpad=15)
        ax.set_zlabel('Característica 3 (Scaled Excentricidad)', labelpad=15)

        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=1)
        plt.close()

    def launch_kmeans(self):
        # Procesamiento de la imagen original
        gray_img, img_filtered, thresh_image, closed_image = self.preprocess_image(self.predict_image)

        # Nos aseguramos que la imagen esté en el formato correcto (uint8)
        if closed_image.dtype == np.bool_:
            closed_image = closed_image.astype(np.uint8) * 255  # Convert boolean to uint8 (0 or 255)

        # Búsqueda de contornos
        contours = self.search_contours(closed_image, min_length=100)

        # Recorta la imagen alrededor del contorno principal
        cropped_image, adjusted_contours, adjusted_hull = self.crop_object(closed_image, contours)

        # Obtiene la imagen que luego se muestra en la interfaz de usuario
        hull_image = self.draw_convex_hull(cropped_image, adjusted_hull)

        # Cálculo de caracteristicas
        predict_features = self.compute_features(cropped_image,adjusted_contours, adjusted_hull)

        # Procesamiento de caracteristicas (cambio de escala)
        predict_features_scaled =  self.scale_features(predict_features)

        # Seleccion de caracteristicas
        predict_features = self.select_features(predict_features)
        predict_features_scaled = self.select_features(predict_features_scaled)

        # Algoritmo Kmeans
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = self.kmeans_plusplus_initialization(self.X, self.k)

            for _ in range(self.max_iters):
                labels = self.assign_clusters(self.X, centroids)
                new_centroids = self.update_centroids(self.X, labels, self.k)

                # Convergencia ?
                if np.linalg.norm(new_centroids - centroids) < self.tol:
                    break

                centroids = new_centroids

            # Cálculo de la inercia del clustering resultante
            inertia = np.sum([np.linalg.norm(self.X[labels == j] - centroids[j]) ** 2 for j in range(self.k)])

            # Actualiza el mejor resultado
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        print(f"Centroides encontrados: {best_centroids}")

        # Cluster to Label mapping
        mapping = self.cluster_to_label_mapping(np.array(self.y), np.array(best_labels))
        print("Mapa Cluster->Clase:", mapping)

        accuracy = self.clustering_accuracy(self.y, best_labels, mapping)
        print(f"Precisión del agrupamiento: {accuracy:.2%}")

        # Predice la imagen
        predicted_group = self.predict(predict_features_scaled.reshape(1, -1), best_centroids)[0]
        predicted_label_int = mapping[predicted_group]
        predicted_label = self.category_mapping[predicted_label_int]
        print(f"Grupo predicho: {predicted_group} - Clase mapeada: {predicted_label_int} - Clase: {predicted_label}")

        if len(predict_features_scaled) == 3:
            self.plot_3d_comparison(predict_features_scaled, self.X, best_labels, '3d_plot_comparison.png')

        return predicted_group, predicted_label, accuracy, hull_image, predict_features, predict_features_scaled, self.chosen_features