import numpy as np
import pandas as pd
import yaml
from itertools import product
import os
import pickle


# Function to pollute dataset for completeness
def pollute_completeness(df, pollution_level, rng):
    pol_df = df.copy()
    total_samples = len(df)
    missing_count = 0

    #remove target col
    target_col = pol_df['classes']
    pol_df = pol_df.drop(columns='classes')
    
    for feature in df.columns:
        missing_count += df[feature].isnull().sum()

    pol_df.fillna(np.nan, inplace=True)
    values_to_pol = int(pollution_level * total_samples) - missing_count
    non_null_indices = pol_df.notnull().values.nonzero()
    indeces_li = list(zip(non_null_indices[0],non_null_indices[1]))
    
    # Randomly choose indices from non-null values
    random_indices = rng.choice(len(non_null_indices[0]), values_to_pol, replace=False)

    # Replace values at randomly chosen indices with NaN
    for idx in random_indices:
        x, y = indeces_li[idx]
        pol_df.iat[x, y] = np.nan


    pol_df['classes'] = target_col.values
    return pol_df

def pollute_feature_accuracy(df, pollution_level, rng):
    pol_df = df.copy()
    total_samples = len(df)

    #remove target col
    target_col = pol_df['classes']
    pol_df = pol_df.drop(columns='classes')
    
    values_to_pol = int(pollution_level * total_samples) 
    non_null_indices = pol_df.notnull().values.nonzero()
    indeces_li = list(zip(non_null_indices[0],non_null_indices[1]))
    
    # Randomly choose indices from non-null values
    random_indices = rng.choice(len(non_null_indices[0]), values_to_pol, replace=False)

    # Replace values at randomly chosen indices with NaN
    for idx in random_indices:
        x, y = indeces_li[idx]
        pol = rng.normal(loc=0, scale=pollution_level)
        pol_df.iat[x, y] = pol_df.iat[x, y] *  pol

    pol_df['classes'] = target_col

    return pol_df

# only works for continues independet variables and class dependent variable
def pollute_uniqueness(df, duplication_factor, rng):
    polluted_df = df.copy()
    for value in df['classes'].unique():
        y = df['classes']
        pollute = df[y == value].values

        # Randomly choose indices from non-null values
        random_indices = rng.choice(len(pollute), duplication_factor-1)

        polution = []
        for idx in random_indices:
            polluted_df.append(pd.DataFrame(polution, index=[idx]))


    return polluted_df


def pollute_df(df,config): 
    rng = np.random.default_rng(config['random_seed'])
    df_all = df.copy()
    df_uni = df.copy()
    df_com = df.copy()
    df_fea = df.copy()


    df_all = pollute_uniqueness(df_all, config['duplicate_factor'], np.random.default_rng(rng.integers(1)))
    df_all = pollute_completeness(df_all, config['completness_pollution'], np.random.default_rng(rng.integers(1)))
    df_all = pollute_feature_accuracy(df_all, config['feature_accuracy_pollution'], np.random.default_rng(rng.integers(1)))

    df_uni = pollute_uniqueness(df_uni, config['duplicate_factor'], np.random.default_rng(rng.integers(1)))
    df_com = pollute_completeness(df_com, config['completness_pollution'], np.random.default_rng(rng.integers(1)))
    df_fea = pollute_feature_accuracy(df_fea, config['feature_accuracy_pollution'], np.random.default_rng(rng.integers(1)))

    return df_all, df_uni, df_com, df_fea



def main(config_file, dataset_dir):
    # Load configuration from YAML fil
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)


    combinations = list(product(*config['pollution_parameters'].values()))


    # Create dictionaries from combinations using two separate loops
    li_param_combination = []
    for combo in combinations:
        temp_dict = {}
        for key, value in zip(config['pollution_parameters'].keys(), combo):
            temp_dict[key] = value
        li_param_combination.append(temp_dict)
  
  
    # Iterate over each file in the folder
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.pkl'):  # Check if file is a pickle file
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, 'rb') as file:
                df = pickle.load(file)
            
            for params in li_param_combination:
                params['random_seed'] = config['random_seed']

                pol_all_df, pol_uni_df, pol_com_df, pol_fea_df = pollute_df(df,params)
                param_string = '_'.join([f"{key}_{value}" for key, value in params.items()])

                pol_all_df.to_pickle(os.path.join(dataset_dir, filename.replace('.pkl', '|') + param_string + ".pkl"))
                pol_uni_df.to_pickle(os.path.join(dataset_dir, filename.replace('.pkl', '|') + 'duplicate_factor_' + str(params['duplicate_factor']) + ".pkl"))
                pol_com_df.to_pickle(os.path.join(dataset_dir, filename.replace('.pkl', '|') + 'completness_pollution_' + str(params['completness_pollution']) + ".pkl"))
                pol_fea_df.to_pickle(os.path.join(dataset_dir, filename.replace('.pkl', '|') + 'feature_accuracy_pollution_' + str(params['feature_accuracy_pollution']) + ".pkl"))

   

if __name__ == "__main__":
    config_file = "/home/simon/uibk/experiment_simon_marinov/pollution_config.yml"
    dataset_dir = '/home/simon/uibk/experiment_simon_marinov/data/train'
    main(config_file, dataset_dir)
