import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    tableau_valeurs_manquantes,
    detecter_XNA,
    creer_colonne_age,
    detecter_et_remplacer_anomalies,
    split_train_test,
    impute_data,
    normalize_data,
    label_encode_binary_columns,
    encode_and_align,
    clean_column_names,
    reintegrate_target
)

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        'DAYS_BIRTH': [-12000, -15000, -20000],
        'DAYS_EMPLOYED': [1000, 365243, 2000],
        'GENDER': ['M', 'XNA', 'F'],
        'TARGET': [0, 1, 0],
        'SK_ID_CURR': [1, 2, 3],
        'CAT': ['A', 'B', 'B']
    })

def test_tableau_valeurs_manquantes():
    df = pd.DataFrame({'col1': [1, None, 3], 'col2': [None, None, 3]})
    result = tableau_valeurs_manquantes(df)
    assert 'col1' in result.index
    assert 'col2' in result.index
    assert result.loc['col2', 'Valeurs manquantes'] == 2

def test_creer_colonne_age(dummy_df):
    df = creer_colonne_age(dummy_df.copy())
    assert 'AGE_YEARS' in df.columns
    assert np.all(df['AGE_YEARS'] > 0)

def test_detecter_et_remplacer_anomalies(dummy_df):
    train, test = detecter_et_remplacer_anomalies(dummy_df.copy(), dummy_df.copy())
    assert 'DAYS_EMPLOYED_ANOM' in train.columns
    assert np.isnan(train.loc[1, 'DAYS_EMPLOYED'])

def test_split_train_test(dummy_df):
    merged_df = dummy_df.copy()
    train_df, test_df = split_train_test(merged_df, dummy_df, dummy_df)
    assert len(train_df) == len(dummy_df)
    assert 'TARGET' not in test_df.columns  # doit être retirée

def test_impute_data():
    train = pd.DataFrame({'A': [1, np.nan, 3], 'TARGET': [0, 1, 1], 'B': ['x', 'x', np.nan]})
    test = pd.DataFrame({'A': [np.nan, 4], 'B': ['x', np.nan]})
    train_imp, test_imp = impute_data(train.copy(), test.copy())
    assert not train_imp.isnull().any().any()
    assert not test_imp.isnull().any().any()

def test_normalize_data():
    train = pd.DataFrame({'A': [1, 2, 3], 'TARGET': [0, 1, 1]})
    test = pd.DataFrame({'A': [4, 5]})
    norm_train, norm_test = normalize_data(train.copy(), test.copy())
    assert norm_train['A'].max() <= 1
    assert norm_train['A'].min() >= 0

def test_label_encode_binary_columns():
    train = pd.DataFrame({'bin': ['yes', 'no', 'yes']})
    test = pd.DataFrame({'bin': ['no', 'yes']})
    train_enc, test_enc = label_encode_binary_columns(train.copy(), test.copy())
    assert pd.api.types.is_integer_dtype(train_enc['bin'])

def test_encode_and_align():
    train = pd.DataFrame({'cat': ['A', 'B'], 'val': [1, 2]})
    test = pd.DataFrame({'cat': ['A', 'C'], 'val': [3, 4]})
    train_enc, test_enc = encode_and_align(train, test)
    assert train_enc.shape[1] == test_enc.shape[1]

def test_clean_column_names():
    df = pd.DataFrame({'A&B': [1], 'C D': [2]})
    df_cleaned = clean_column_names(df.copy())
    assert 'A_B' in df_cleaned.columns
    assert 'C_D' in df_cleaned.columns

def test_reintegrate_target():
    df = pd.DataFrame({'A': [1, 2, 3]})
    target = pd.Series([0, 1, 0])
    df_new = reintegrate_target(df.copy(), target)
    assert 'TARGET' in df_new.columns
    assert (df_new['TARGET'] == target).all()