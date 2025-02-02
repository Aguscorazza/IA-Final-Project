from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QFileDialog, QRadioButton, \
    QButtonGroup, QGridLayout, QSizePolicy
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np


class ObjectRecognitionWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.setWindowTitle("Object Recognition")
        self.setMinimumSize(600, 800)
        self.setStyleSheet("background-color: none;")

        self.controller = controller
        # Store image array
        self.image_array = None
        self.processed_images = []  # Store processed images as numpy arrays


        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)

        # Left column layout
        left_column = QVBoxLayout()
        left_column.setAlignment(Qt.AlignTop)
        left_column.setAlignment(Qt.AlignHCenter)

        # Add widgets to the left column
        label = QLabel("Object Recognition Window")
        label.setStyleSheet("font-size: 18pt; padding: 10px; font-weight: bold;")
        left_column.addWidget(label)

        # Button to load image
        self.load_button = QPushButton("Load Image")
        self.load_button.setStyleSheet("font-size: 12pt; padding: 5px; background-color: rgb(200, 200, 200);")
        self.load_button.clicked.connect(self.load_image)  # Connect to function
        left_column.addWidget(self.load_button)

        # Image display label
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black; padding: 5px;")
        left_column.addWidget(self.image_label)

        # Recognition algorithm label
        self.algorithm_label = QLabel("Recognition algorithm:")
        self.algorithm_label.setAlignment(Qt.AlignLeft)
        self.algorithm_label.setStyleSheet("padding: 5px; font-size: 12pt;")
        left_column.addWidget(self.algorithm_label)

        # Horizontal layout for radio buttons
        algo_layout = QHBoxLayout()
        algo_layout.setAlignment(Qt.AlignHCenter)

        # KMeans radio button
        self.kmeans_radio = QRadioButton("K-Means")
        self.kmeans_radio.setStyleSheet("padding-right: 30%; font-size: 10pt;")
        self.kmeans_radio.setChecked(True)  # Default selection
        algo_layout.addWidget(self.kmeans_radio)

        # KNN radio button
        self.knn_radio = QRadioButton("KNN")
        self.knn_radio.setStyleSheet("padding-left: 30%; font-size: 10pt;")
        algo_layout.addWidget(self.knn_radio)

        # Group radio buttons
        self.algo_group = QButtonGroup(self)
        self.algo_group.addButton(self.kmeans_radio)
        self.algo_group.addButton(self.knn_radio)

        left_column.addLayout(algo_layout)

        self.launch = QPushButton("Launch algorithm")
        self.launch.setStyleSheet("font-size: 12pt; padding: 5px; margin:10px 50px; background-color: rgb(200, 200, 200);")
        self.launch.clicked.connect(self.run_algorithm)
        left_column.addWidget(self.launch)

        # **Vote Result Label**
        self.vote_result_label = QLabel("")
        self.vote_result_label.setAlignment(Qt.AlignCenter)
        self.vote_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: blue; padding-top: 10px;")
        left_column.addWidget(self.vote_result_label)

        # **Confidence Label**
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green; padding-bottom: 10px;")
        left_column.addWidget(self.confidence_label)

        main_layout.addLayout(left_column, 1)

        ### RIGHT COLUMN (PROCESSED IMAGES) ###
        # Right column layout
        right_column = QVBoxLayout()
        right_column.setAlignment(Qt.AlignTop)

        # Label for preprocessed image
        self.preprocessed_image_text = QLabel("Preprocessed Image with Convex Hull")
        self.preprocessed_image_text.setVisible(False)
        self.preprocessed_image_text.setAlignment(Qt.AlignCenter)
        self.preprocessed_image_text.setStyleSheet("padding-top:30px; font-size: 14pt")

        self.processed_image_label = QLabel("")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setStyleSheet("padding:5px 50px")

        right_column.addWidget(self.preprocessed_image_text)
        right_column.addWidget(self.processed_image_label)


        # Label for features title
        self.features_label = QLabel("Preprocessed Features")
        self.features_label.setAlignment(Qt.AlignHCenter)
        self.features_label.setStyleSheet("font-weight: bold; font-size:12pt; padding:0px;")
        self.features_label.setVisible(False)
        right_column.addWidget(self.features_label)

        # Grid layout for displaying features (3 rows × 3 columns)
        self.features_grid = QGridLayout()
        self.features_grid.setAlignment(Qt.AlignHCenter)
        # Add layout for displaying preprocessed features
        self.features_container = QWidget()
        self.features_container.setStyleSheet("padding: 0px;")
        self.features_container.setLayout(self.features_grid)
        right_column.addWidget(self.features_container)

        main_layout.addLayout(right_column, 3)  # Right column takes more space

        self.setLayout(main_layout)

    def load_image(self):
        """ Open file dialog to select an image and display it. """
        self.reset_labels()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options
        )
        if file_path:
            # Load image with PIL
            pil_image = Image.open(file_path)
            # Convert to NumPy array and store it
            self.image_array = np.array(pil_image)

            # Display the image in the QLabel
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.image_label.setText("")  # Clear placeholder text

    def run_algorithm(self):
        """ Send the image array to the controller when launch button is clicked. """
        if self.image_array is None:
            self.image_label.setText("Please load an image first.")
            return

        # Determine selected algorithm
        algorithm = "K-Means" if self.kmeans_radio.isChecked() else "KNN"

        # Call controller function with image and selected algorithm
        result_list = self.controller.launch_algorithm(algorithm, self.image_array)


        if len(result_list) == 5:
            # KNN results
            vote_result, confidence, processed_image, features, scaled_features = result_list

            # **Update Labels**
            self.vote_result_label.setText(f"Vote Result: {vote_result}")
            self.confidence_label.setText(f"Confidence: {confidence * 100:.2f}%")
        else:
            vote_group, vote_result, confidence, processed_image, features, scaled_features = result_list
            self.vote_result_label.setText(f"Vote Group: {vote_group} - Group Label: {vote_result}")
            self.confidence_label.setText(f"Clustering accuracy: {confidence * 100:.2f}%")

        # Display Preprocessed Images
        self.update_preprocessed_image(processed_image)
        self.update_preprocessed_features(features, scaled_features)

    def update_preprocessed_image(self, image_array):
        """
        Update the QLabel with the new preprocessed image.

        Parameters:
            image_array (np.array): The processed image to display.
        """
        if image_array is not None:
            # Convert NumPy array to QImage
            height, width, channels = image_array.shape
            bytes_per_line = channels * width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap and update QLabel with smaller size
            pixmap = QPixmap.fromImage(q_image)
            self.preprocessed_image_text.setVisible(True)
            self.processed_image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))  # Scale down to 200x200
            self.processed_image_label.setVisible(True)

    def update_preprocessed_features(self, features, scaled_features):
        """
        Update the grid of preprocessed features in the right column.

        Parameters:
            features (dict or list): Preprocessed features to display.
        """
        # Assuming features is a dictionary with keys and values
        row = 0
        col = 0
        self.features_label.setVisible(True)
        if isinstance(features, dict):
            for feature_name, feature_value in features.items():
                # Add feature name as QLabel
                feature_name_label = QLabel(f"{feature_name}: {feature_value:.2f}")
                feature_name_label.setStyleSheet("padding: 5px 20px; font-size: 12pt;")
                self.features_grid.addWidget(feature_name_label, row, col)

                row += 1

        row = 0
        col = 1
        if isinstance(features, dict):
            for feature_name, feature_value in scaled_features.items():
                # Add feature name as QLabel
                feature_name_label = QLabel(f"{feature_name}: {feature_value:.2f}")
                feature_name_label.setStyleSheet("padding: 5px 20px; font-size: 12pt;")
                self.features_grid.addWidget(feature_name_label, row, col)

                row += 1

    def reset_labels(self):
        for i in reversed(range(self.features_grid.count())):
            widget = self.features_grid.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(False)
                widget.deleteLater()

        self.vote_result_label.setText("")
        self.confidence_label.setText("")
        self.features_label.setVisible(False)
        self.preprocessed_image_text.setVisible(False)
        self.processed_image_label.setVisible(False)

