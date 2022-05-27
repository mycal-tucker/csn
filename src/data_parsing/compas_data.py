# Taken from https://github.com/PAIR-code/what-if-tool/blob/master/WIT_COMPAS_with_SHAP.ipynb
import numpy as np
import pandas as pd


def get_compas_data():
    df = pd.read_csv('https://storage.googleapis.com/what-if-tool-resources/computefest2019/cox-violent-parsed_filt.csv')

    # Preprocess the data

    # Filter out entries with no indication of recidivism or no compass score
    df = df[df['is_recid'] != -1]
    df = df[df['decile_score'] != -1]

    # Rename recidivism column
    df['recidivism_within_2_years'] = df['is_recid']

    # Make the COMPASS label column numeric (0 and 1), for use in our model
    df['COMPASS_determination'] = np.where(df['score_text'] == 'Low', 0, 1)

    df = pd.get_dummies(df, columns=['sex', 'race'])

    # Get list of all columns from the dataset we will use for model input or output.
    input_features = ['sex_Female', 'sex_Male', 'age', 'race_African-American', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Other', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']

    to_keep = input_features + ['recidivism_within_2_years', 'COMPASS_determination']

    to_remove = [col for col in df.columns if col not in to_keep]
    df = df.drop(columns=to_remove)
    # normalize data by columns
    df = (df - df.min()) / (df.max() - df.min())

    input_columns = df.columns.tolist()
    print(df.head())
    labels = df['COMPASS_determination']
    # Choose a protected field to also predict,
    # protected_label = df['sex_Female'].values
    # protected_label = df['race_African-American'].values
    # protected_label = df['recidivism_within_2_years'].values
    protected_field_names = ['race_African-American', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Other']
    race_data = df[protected_field_names]
    race_data_np = race_data.values
    protected_indices = np.argmax(race_data_np, axis=1)
    protected_label = protected_indices

    # Create data structures needing for training and testing.
    # The training data doesn't contain the column we are predicting,
    # 'COMPASS_determination', or the column we are using for evaluation of our
    # trained model, 'recidivism_within_2_years'.
    df_for_training = df.drop(columns=['COMPASS_determination', 'recidivism_within_2_years'])
    df_for_training = df_for_training.drop(columns=protected_field_names)
    train_size = int(len(df_for_training) * 0.8)

    train_data = df_for_training[:train_size]
    train_labels = labels[:train_size]
    train_protected = protected_label[:train_size]

    test_data = df_for_training[train_size:]
    test_labels = labels[train_size:]
    test_protected = protected_label[train_size:]

    # Return the data, converting everything as needed into numpy arrays instead of data frames.
    return train_data.values, train_labels.values, train_protected, test_data.values, test_labels.values, test_protected


