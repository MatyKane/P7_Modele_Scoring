def tableau_valeurs_manquantes(df):
    """Calcule et affiche un tableau des valeurs manquantes par colonne avec leurs pourcentages."""
    nb_manquantes = df.isnull().sum()
    pourcentage_manquantes = 100 * nb_manquantes / len(df)
    tableau = pd.concat([nb_manquantes, pourcentage_manquantes], axis=1)
    tableau.columns = ['Valeurs manquantes', '% du total']
    tableau = tableau[tableau['Valeurs manquantes'] > 0]
    tableau = tableau.sort_values('% du total', ascending=False).round(1)
    print(f"Le dataframe sélectionné contient {df.shape[1]} colonnes.\n"
          f"{tableau.shape[0]} colonnes ont des valeurs manquantes.")
    return tableau


def detecter_XNA(df):
    """Affiche les colonnes contenant la valeur 'XNA' dans les colonnes de type object."""
    for col in df.columns:
        if df[col].dtype == 'object':
            if 'XNA' in df[col].unique():
                print(f'"XNA" trouvé dans la colonne : {col} (Nombre : {sum(df[col] == "XNA")})')



def creer_colonne_age(app_df):
    """Crée la colonne 'AGE_YEARS' à partir de 'DAYS_BIRTH' (valeurs négatives)."""
    app_df['AGE_YEARS'] = -app_df['DAYS_BIRTH'] / 365
    return app_df


def detecter_et_remplacer_anomalies(app_train, app_test):
    """Détecte les anomalies dans DAYS_EMPLOYED et crée un indicateur, puis remplace les valeurs anormales par NaN."""
    anomalie_valeur = 365243
    for df in [app_train, app_test]:
        df['DAYS_EMPLOYED_ANOM'] = df['DAYS_EMPLOYED'] == anomalie_valeur
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({anomalie_valeur: np.nan})
    return app_train, app_test


def corr_target(app_train):
    """Affiche les 10 corrélations positives et négatives les plus fortes avec la variable TARGET."""
    correlations = app_train.corr(numeric_only=True)['TARGET'].sort_values()
    print("Corrélations positives les plus fortes :\n", correlations.tail(10))
    print("\nCorrélations négatives les plus fortes :\n", correlations.head(10))


def bin_age(app_train):
    """Crée une colonne de bins d'âge et affiche les premières lignes."""
    age_data = app_train[['TARGET', 'DAYS_BIRTH', 'AGE_YEARS']].copy()
    age_data['AGE_BINNED'] = pd.cut(age_data['AGE_YEARS'], bins=np.linspace(20, 70, num=11))
    print(age_data.head(10))
    age_groups = age_data.groupby('AGE_BINNED', observed=False).mean()
    return age_groups


def split_train_test(data, app_train, app_test):
    data_train = data[data['SK_ID_CURR'].isin(app_train.SK_ID_CURR)]
    data_test = data[data['SK_ID_CURR'].isin(app_test.SK_ID_CURR)]
    data_test = data_test.drop('TARGET', axis=1, errors='ignore')
    return data_train, data_test

def impute_data(data_train, data_test):
    from sklearn.impute import SimpleImputer

    # Imputation numérique
    num_cols = data_train.select_dtypes(include=['int64', 'float64']).columns.drop('TARGET', errors='ignore')
    imputer = SimpleImputer(strategy='median')
    data_train[num_cols] = imputer.fit_transform(data_train[num_cols])
    data_test[num_cols] = imputer.transform(data_test[num_cols])

    # Imputation catégorielle
    cat_cols = data_train.select_dtypes(include=['object', 'category']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data_train[cat_cols] = imputer_cat.fit_transform(data_train[cat_cols])
    data_test[cat_cols] = imputer_cat.transform(data_test[cat_cols])

    return data_train, data_test

def normalize_data(data_train, data_test):
    from sklearn.preprocessing import MinMaxScaler

    num_cols = data_train.select_dtypes(include=['int64', 'float64']).columns.drop('TARGET', errors='ignore')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train[num_cols])

    data_train[num_cols] = scaler.transform(data_train[num_cols])
    data_test[num_cols] = scaler.transform(data_test[num_cols])

    return data_train, data_test

def label_encode_binary_columns(data_train, data_test):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    for col in data_train.columns:
        if data_train[col].dtype == 'object' and len(data_train[col].unique()) <= 2:
            le.fit(data_train[col])
            data_train[col] = le.transform(data_train[col])
            data_test[col] = le.transform(data_test[col])
    return data_train, data_test

def encode_and_align(data_train, data_test):
    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)
    data_train, data_test = data_train.align(data_test, join='inner', axis=1)
    return data_train, data_test

def clean_column_names(df):
    import re
    rename_mapping = {}
    for col in df.columns:
        new_col = re.sub(r'[^0-9a-zA-Z_]', '_', col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        rename_mapping[col] = new_col
    df.rename(columns=rename_mapping, inplace=True)
    return df

def reintegrate_target(data_train, target):
    data_train['TARGET'] = target
    return data_train