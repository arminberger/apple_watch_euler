import sys
import os
import data_cleaning
import training

def get_paths():
    try:
        project_root_path = sys.argv[1]
        data_path = sys.argv[2]
        model_path = sys.argv[3]
    except IndexError:
        print('Please provide the path to the project root, data folder and model (in this order)')
        return None, None

    if not os.path.exists(project_root_path):
        print('The path to the project root does not exist')
        return None, None
    if not os.path.exists(data_path):
        print('The path to the data folder does not exist')
        return None, None
    if not os.path.exists(model_path):
        print('The path to the model folder does not exist')
        return None, None

    print(project_root_path)
    print(data_path)
    return project_root_path, data_path

def main():
    project_root_path, data_path = get_paths()
    if project_root_path is None or data_path is None:
        return

    # Do something with the paths
    data_cleaning.clean_data(project_root_path, data_path)
    training.train_model(project_root_path, data_path, model_path)

if __name__ == '__main__':
    main()




