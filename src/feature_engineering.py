def ajouter_features_et_correlation(data):
    """
    Ajoute de nouvelles variables issues de l'ingénierie de features,
    calcule la matrice de corrélation et affiche les plus fortes corrélations avec la variable TARGET.
    
    Paramètre :
    ----------
    data : pd.DataFrame
        Le dataframe contenant les colonnes nécessaires.

    Retour :
    -------
    data : pd.DataFrame
        Le dataframe enrichi avec les nouvelles colonnes d'ingénierie.
    """
    # Création de nouvelles variables
    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']

    # Variables créées
    features_engin = [
        'PREVIOUS_LOANS_COUNT',
        'MONTHS_BALANCE_MEAN',
        'PREVIOUS_APPLICATION_COUNT',
        'CREDIT_INCOME_PERCENT',
        'ANNUITY_INCOME_PERCENT',
        'CREDIT_TERM',
        'DAYS_EMPLOYED_PERCENT'
    ]

    # Calcul des corrélations numériques
    correlations = data.select_dtypes(include='number').corr()

    # Affichage des corrélations avec TARGET
    print(' Most Positive Correlations with TARGET:\n')
    print(correlations['TARGET'].sort_values(ascending=False).head(15))

    print('\n Most Negative Correlations with TARGET:\n')
    print(correlations['TARGET'].sort_values().head(15))

    return data