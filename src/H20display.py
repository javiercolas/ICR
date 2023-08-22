import os
import json


class H20Display:
    def __init__(self, evaluate_info, X_columns, output_path='data/results', filename='results.json'):

        self.evaluate_info = evaluate_info
        self.X_columns = X_columns
        self.output_path = output_path
        self.filename = filename

    def display(self):
        self._check_and_create_json_file()
        self._update_json_file()

    def _check_and_create_json_file(self):
        file_path = os.path.join(self.output_path, self.filename)
        if not os.path.isfile(file_path):
            try:
                with open(file_path, 'w') as file:
                    json.dump({}, file)
                print(f"The JSON file '{file_path}' has been created.")
            except IOError as e:
                print(f"Error creating JSON file '{file_path}': {str(e)}")

    def _update_json_file(self):
        file_path = os.path.join(self.output_path, self.filename)
        try:
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
            existing_data[f"model_{'_'.join(self.X_columns)}"] = self.evaluate_info
            with open(file_path, 'w') as file:
                json.dump(existing_data, file, indent=4)
        except IOError as e:
            print(f"Error updating JSON file '{file_path}': {str(e)}")

