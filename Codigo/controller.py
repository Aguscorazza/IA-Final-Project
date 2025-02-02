import subprocess
from ObjectRecognitionAlgorithm import KNN, KMeans

class MazeController:
    def __init__(self, maze):
        self.maze = maze
        self.solver = None
        self.maze_widget = None  # Initialize it later

        # PDDL parameters
        self.problem_name = "BLOCKS-4"
        self.domain_name = "STRIPS_domain"
        self.objects = "WASHER NUT NAIL BOLT"
        self.problem_file = "test_problem.pddl"


    def set_solver(self, solver):
        self.solver = solver
        self.solver.controller = self  # Link solver to controller

    def solve_maze(self):
        if self.solver:
            self.solver.solve()

    def generate_maze(self):
        if self.maze:
            self.maze.generateMaze()

    def clear_maze(self):
        if self.maze:
            self.maze.clearMaze()

    """
    STRIPS Box Order Functionalities
    """

    @staticmethod
    def get_rectangles_order(rectangles):
        rectangles_order = [(index, rectangle["label"].upper()) for index,rectangle in enumerate(rectangles)]
        return  rectangles_order


    def write_pddl_problem_file(self, rectangles_init, rectangles_goal):
        init_order = self.get_rectangles_order(rectangles_init)
        goal_order = self.get_rectangles_order(rectangles_goal)

        init_string = f"""(HANDEMPTY)
(CLEAR {init_order[0][1]})
(ONTABLE {init_order[3][1]})
(ON {init_order[0][1]} {init_order[1][1]})
(ON {init_order[1][1]} {init_order[2][1]})
(ON {init_order[2][1]} {init_order[3][1]})
"""

        goal_string = f"""(HANDEMPTY)
(CLEAR {goal_order[0][1]})
(ONTABLE {goal_order[3][1]})
(ON {goal_order[0][1]} {goal_order[1][1]})
(ON {goal_order[1][1]} {goal_order[2][1]})
(ON {goal_order[2][1]} {goal_order[3][1]})
"""
        #print(init_string)
        #print(goal_string)

        problem_content = f"""(define 
(problem {self.problem_name})
(:domain {self.domain_name})
(:objects {"".join(self.objects)})
(:init {"".join(init_string)})
(:goal (and {"".join(goal_string)}))
)
"""
        #print((problem_content))
        try:
            with open(self.problem_file, 'w') as file:
                file.write(problem_content)
                print(f"Problem file '{self.problem_file}' created successfully.")
                return self.problem_file
        except:
            return None


    def search_ppdl_solution(self, rectangles_init, rectangles_goal):
        filename = self.write_pddl_problem_file(rectangles_init, rectangles_goal)
        if filename is None:
            raise "Couldn't create PPDL problem file."

        def run_planner(problem_file, search_algorithm="astar(blind())"):
            """
            Runs the Fast Downward planner with the specified search algorithm.
            """
            root = 'downward'
            domain_file = "STRIPS_domain.pddl"
            try:
                # Command to run Fast Downward
                command = [
                    "python",
                    f"{root}/fast-downward.py",  # Path to the Fast Downward script
                    domain_file,
                    problem_file,
                    "--search", search_algorithm
                ]

                # Execute the command and capture the output
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                # Print the result's stdout (planner output)
                #print("Planner output:")
                #print(result.stdout)
                return result.stdout
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e.stderr}")
            except FileNotFoundError:
                print(
                    "Fast Downward executable not found. Make sure you've built Fast Downward and set the correct path.")

        planning_result = run_planner(problem_file=filename, search_algorithm="astar(lmcut())")
        return self.extract_actions(planning_result)

    @staticmethod
    def extract_actions(log):
        # Define action keywords to look for
        action_keywords = ["unstack", "put-down", "pick-up", "stack"]
        actions = []

        # Split the log into lines and check each line
        for line in log.splitlines():
            if any(line.startswith(keyword) for keyword in action_keywords):
                actions.append(line.strip())  # Add the action to the list

        return actions

    """Object Recognition Functionalities"""
    def launch_algorithm(self, algorithm, image):
        if algorithm == "K-Means":
            # Calls KMeans
            kmeans = KMeans(train_filename='train_features.csv', predict_image=image)
            vote_group, vote_result, confidence, hull_image, features, scaled_features = kmeans.launch_kmeans()

            features_names = ['Hu_Moment_1', 'Circle_Area_Ratio', 'Eccentricity']
            features_dict = dict(zip(features_names, features))
            scaled_features_names = ['Scaled_Hu_Moment_1', 'Scaled_Circle_Area_Ratio', 'Scaled_Eccentricity']
            features_dict_scaled = dict(zip(scaled_features_names, scaled_features))

            return [vote_group, vote_result, confidence, hull_image, features_dict, features_dict_scaled]

        elif algorithm == "KNN":
            # Calls KNN
            knn = KNN(k=7, train_filename='train_features.csv', predict_image=image)
            vote_result, confidence, hull_image, features, scaled_features = knn.launch_knn()

            features_names = ['Hu_Moment_1', 'Circle_Area_Ratio', 'Eccentricity']
            features_dict = dict(zip(features_names, features))
            scaled_features_names = ['Scaled_Hu_Moment_1', 'Scaled_Circle_Area_Ratio', 'Scaled_Eccentricity']
            features_dict_scaled = dict(zip(scaled_features_names, scaled_features))

            return [vote_result, confidence, hull_image, features_dict, features_dict_scaled]

        else:
            raise "Invalid algorithm."



