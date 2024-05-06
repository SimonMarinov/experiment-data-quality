import yaml
import pandas as pd
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from itertools import product
import os
from sklearn.model_selection import train_test_split
from sklearn import datasets


# Activate automatic conversion of R objects to pandas dataframes
pandas2ri.activate()

# Import the mlbench package from R
mlbench = importr('mlbench')

def generate_dataset(mlbench_function, args):
    """
    Generate dataset using mlbench functions.
 
    """
    dataset = mlbench_function(*args)
    r_dataframe = ro.r['data.frame'](dataset)

    # Convert R dataframe to pandas DataFrame
    df = pandas2ri.rpy2py_dataframe(r_dataframe)
    return df

def generate_datasets(config):
    dataset_li = []

    for dataset_config in config['datasets']:
        dataset_name = dataset_config['mlbench_function']
        mlbench_function_name = dataset_config['mlbench_function']
        mlbench_function = getattr(mlbench, mlbench_function_name)

        params = dataset_config['params']
    
        param_combinations = list(product(*params.values()))

        for combination in param_combinations:

            # Generate dataset
            dataset_df = generate_dataset(mlbench_function, combination)
            
            dataset_li.append(dataset_df)


    return dataset_li

def safe_df(df, config, train_data_dir, test_data_dir, dataset_name, param_string=''):

    train_test_split_args = config['settings']['train_test_split_args']
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='classes'),
                                                            df['classes'],
                                                            **train_test_split_args)


    train_df = X_train.join(y_train)
    test_df = X_test.join(y_test)
    
    train_df.to_pickle(os.path.join(dataset_dir, train_data_dir,
                                    dataset_name + param_string +".pkl"))

    
    test_df.to_pickle(os.path.join(dataset_dir, test_data_dir, 
                                    dataset_name + param_string +".pkl"))


#safe the dataset into_dataset
    

def main(config_file, dataset_dir, test_data_dir, train_data_dir):
    prefix = 'mlbench_'

    # Load configuration from YAML file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    for dataset_config in config['mlbench_datasets']:
        dataset_name = dataset_config['mlbench_function']
        mlbench_function_name = prefix + dataset_config['mlbench_function'] 
        mlbench_function = getattr(mlbench, mlbench_function_name)

        params = dataset_config['params']
    
        param_combinations = list(product(*params.values()))

        for combination in param_combinations:

            param_string = '_'.join([f"{key}_{value}" for key, value in zip(params.keys(), combination)])

            # Generate dataset
            df = generate_dataset(mlbench_function, combination)

            safe_df(df, config, train_data_dir, test_data_dir, dataset_name, param_string)

    for dataset_name, save_name in config['sklearn_datasets'].items():
        try:
            dataset = getattr(datasets, dataset_name)()
        except AttributeError:
            print(f"Dataset '{dataset_name}' not found in sklearn.datasets.")
            continue
        
        df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        df['classes'] = dataset.target


        safe_df(df, config, train_data_dir, test_data_dir, save_name)




if __name__ == "__main__":
    config_file = "data_config.yml"  # Specify your YAML configuration file here
    dataset_dir = 'data/'
    main(config_file, dataset_dir, 'test', 'train')
