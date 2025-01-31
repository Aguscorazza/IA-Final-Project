from PyQt5.QtCore import QRect, Qt, QSize, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QImage, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QPushButton


class BoxOrderWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Box Order")
        self.setMinimumSize(600, 800)
        self.setStyleSheet("background-color: none;")

        title_font = QFont("Arial", 15)
        normal_font = QFont("Arial", 10)

        # Create the main vertical layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        top_layout = QHBoxLayout()
        buttons_column_layout = QVBoxLayout()
        column1_layout = QVBoxLayout()
        column2_layout = QVBoxLayout()

        column_width = 50  # You can adjust this value to your desired column width

        # Create spacers with the desired width
        spacer_left = QSpacerItem(column_width, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)
        spacer_right = QSpacerItem(column_width, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)

        # Add the spacers and the column layouts to the top layout
        top_layout.addSpacerItem(spacer_left)
        top_layout.addLayout(buttons_column_layout)
        top_layout.addLayout(column1_layout)
        top_layout.addLayout(column2_layout)
        top_layout.addSpacerItem(spacer_right)

        main_layout.addLayout(top_layout)
        top_layout.setAlignment(column1_layout, Qt.AlignLeft)
        top_layout.setAlignment(column2_layout, Qt.AlignRight)

        self.title_label = QLabel("Blocks world Solver")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.title_label.setFont(title_font)
        self.title_label.setContentsMargins(0, 20, 0, 10)

        self.search_button = QPushButton("Search Solution")
        self.search_button.setFont(normal_font)
        self.search_button.setStyleSheet("background-color: #CCD2CD;")
        self.search_button.setFixedSize(250, 40)

        buttons_column_layout.addWidget(self.title_label)
        buttons_column_layout.addWidget(self.search_button)

        # Create the label under the search button (Initially hidden)
        self.search_result_label = QLabel("", self)
        self.search_result_label.setFont(normal_font)
        #self.search_result_label.setAlignment(Qt.AlignCenter)
        self.search_result_label.setStyleSheet("color: black; font-size: 20px; padding-top: 20px")
        self.search_result_label.setVisible(True)  # Hide initially
        self.search_result_label.setContentsMargins(0, 500, 0, 10)

        buttons_column_layout.addWidget(self.search_result_label)

        buttons_column_layout.addStretch()  # Push the button to the top of the column

        # Create and add the global label
        self.global_label = QLabel("Initial State", self)
        self.global_label.setAlignment(Qt.AlignLeft)
        self.global_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px; padding-left: 70px;")
        self.global_label.setFixedHeight(40)  # Set a fixed height for the label
        column1_layout.addWidget(self.global_label)

        # Spacer to push the rectangles down below the global label
        column1_layout.addStretch()

        self.final_state_label = QLabel("Final State", self)
        self.final_state_label.setAlignment(Qt.AlignLeft)
        self.final_state_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px; padding-right: 420px;")
        self.final_state_label.setFixedHeight(40)
        column2_layout.addWidget(self.final_state_label)

        column2_layout.addStretch()

        # Create a vertical layout for the rectangles
        self.rectangles_init_layout = QVBoxLayout()
        self.rectangles_final_layout = QVBoxLayout()

        column1_layout.addLayout(self.rectangles_init_layout)
        column2_layout.addLayout(self.rectangles_final_layout)

        # Define initial rectangle positions and images
        self.rect_pos_init = {
            "#2A9D8F": (590, 50, 400, 200),
            "#E9C46A": (590, 250, 400, 200),
            "#F4A261": (590, 450, 400, 200),
            "#E76F51": (590, 650, 400, 200),
        }

        self.rect_pos_final = {
            "#2A9D8F": (1180, 50, 400, 200),
            "#E9C46A": (1180, 250, 400, 200),
            "#F4A261": (1180, 450, 400, 200),
            "#E76F51": (1180, 650, 400, 200),
        }

        self.images = [
            r"images\icons\bolt-icon.png",
            r"images\icons\nut.svg",
            r"images\icons\washer.png",
            r"images\icons\nail-icon.png",
        ]

        self.labels = [
            "Bolt",
            "Nut",
            "Washer",
            "Nail"
        ]

        self.rectangles_init = [
            {"color": QColor(color), "rect": QRect(*pos), "index": i, "image": QImage(image_path), "label": label}
            for i, (color, (pos, image_path, label)) in
            enumerate(zip(self.rect_pos_init.keys(), zip(self.rect_pos_init.values(), self.images, self.labels)))
        ]

        self.rectangles_final = [
            {"color": QColor(color), "rect": QRect(*pos), "index": i, "image": QImage(image_path), "label": label}
            for i, (color, (pos, image_path, label)) in
            enumerate(zip(self.rect_pos_final.keys(), zip(self.rect_pos_final.values(), self.images, self.labels)))
        ]

        self.dragging_index = None
        self.dragging_column = None  # To track the column being dragged
        self.start_drag_pos = None

        # Add the rectangles to the rectangles layout
        self.add_rectangles_to_layout()

        self.search_button.clicked.connect(self.on_search_clicked)

    def on_search_clicked(self):
        """Call the controller's get_rectangles_order function when the button is clicked."""
        solution = self.controller.search_ppdl_solution(self.rectangles_init, self.rectangles_final)

        # Display solution in a label under the search button
        solution_text = "Found solution : \n"
        #print(solution)
        for i, step in enumerate(solution):
            solution_text += f"\t{i}): {step}\n"
        #print(solution_text)
        self.search_result_label.setText(solution_text)
        self.search_result_label.setVisible(True)

    def add_rectangles_to_layout(self):
        for rect_info in self.rectangles_init:
            # Create a widget for each rectangle and add it to the rectangles layout
            rect_widget = QWidget()
            rect_widget.setFixedSize(rect_info["rect"].size())
            self.rectangles_init_layout.addWidget(rect_widget)

        for rect_info in self.rectangles_final:
            rect_widget = QWidget()
            rect_widget.setFixedSize(rect_info["rect"].size())
            self.rectangles_final_layout.addWidget(rect_widget)

    def paintEvent(self, event):
        painter = QPainter(self)
        font = QFont("Arial", 12)
        painter.setFont(font)

        for rect_info in self.rectangles_init:
            painter.setBrush(QBrush(rect_info["color"]))
            painter.drawRect(rect_info["rect"])

            # Draw the image inside the rectangle
            image_rect = rect_info["rect"]
            image = rect_info["image"]
            if not image.isNull():
                # Define the size of the image to be a small portion of the rectangle
                image_size = QSize(int(image_rect.width() / 1.5), int(image_rect.height() / 1.5))
                scaled_image = image.scaled(image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Calculate the position to center the image within the rectangle
                x_offset = (image_rect.width() - scaled_image.width()) // 2
                y_offset = (image_rect.height() - scaled_image.height()) // 2
                image_position = image_rect.topLeft() + QPoint(x_offset, y_offset)

                # Draw the image in the center of the rectangle
                painter.drawImage(QRect(image_position, scaled_image.size()), scaled_image)

        for rect_info in self.rectangles_final:
            painter.setBrush(QBrush(rect_info["color"]))
            painter.drawRect(rect_info["rect"])

            # Draw the label at the top of the rectangle
            label_rect = rect_info["rect"].adjusted(0, -20, 0, 0)  # Adjusted to move label above rectangle
            painter.setPen(QColor("black"))

            # Draw the image inside the rectangle
            image_rect = rect_info["rect"]
            image = rect_info["image"]
            if not image.isNull():
                # Define the size of the image to be a small portion of the rectangle
                image_size = QSize(int(image_rect.width() / 1.5), int(image_rect.height() / 1.5))
                scaled_image = image.scaled(image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Calculate the position to center the image within the rectangle
                x_offset = (image_rect.width() - scaled_image.width()) // 2
                y_offset = (image_rect.height() - scaled_image.height()) // 2
                image_position = image_rect.topLeft() + QPoint(x_offset, y_offset)

                # Draw the image in the center of the rectangle
                painter.drawImage(QRect(image_position, scaled_image.size()), scaled_image)

    def mousePressEvent(self, event):
        for i, rect_info in enumerate(self.rectangles_init):
            if rect_info["rect"].contains(event.pos()):
                self.dragging_index = i
                self.dragging_column = 1
                self.start_drag_pos = event.pos()
                return

        for i, rect_info in enumerate(self.rectangles_final):
            if rect_info["rect"].contains(event.pos()):
                self.dragging_index = i
                self.dragging_column = 2
                self.start_drag_pos = event.pos()
                return

    def mouseMoveEvent(self, event):
        if self.dragging_index is not None:
            mouse_y = event.y()
            if self.dragging_column == 1:
                rectangles = self.rectangles_init
            else:
                rectangles = self.rectangles_final

            y_inferior = [rect["rect"].y() + rect["rect"].height() for rect in rectangles]

            #print(self.controller.get_rectangles_order(rectangles))

            # Determine the new position for the dragged rectangle
            new_index = None
            for i, y in enumerate(y_inferior):
                if mouse_y < y:
                    new_index = i
                    break

            # If a new position is found and it's different from the current position
            if new_index is not None and new_index != self.dragging_index:
                # Move the dragged rectangle to the new position
                rect_to_move = rectangles.pop(self.dragging_index)
                rectangles.insert(new_index, rect_to_move)
                # Swap the positions of rectangles to reflect the new order
                if new_index < self.dragging_index:
                    # If the new index is smaller, move rectangles up
                    for i in range(self.dragging_index - 1, new_index - 1, -1):
                        rectangles[i]["rect"], rectangles[i + 1]["rect"] = rectangles[i + 1]["rect"], rectangles[i]["rect"]
                else:
                    # If the new index is larger, move rectangles down
                    for i in range(self.dragging_index + 1, new_index + 1):
                        rectangles[i]["rect"], rectangles[i - 1]["rect"] = rectangles[i - 1]["rect"], rectangles[i]["rect"]

                self.dragging_index = new_index

            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging_index is not None:
            # Reorder rectangles based on their new positions
            if self.dragging_column == 1:
                self.rectangles_init.sort(key=lambda r: r["rect"].top())
            else:
                self.rectangles_final.sort(key=lambda r: r["rect"].top())
            self.dragging_index = None
            self.update()
