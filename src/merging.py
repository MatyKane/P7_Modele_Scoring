def fusion_train_bureau(app_train, bureau):
    """Fusionne le train avec bureau selon SK_ID_CURR."""
    return app_train.merge(bureau, on='SK_ID_CURR', how='inner')


def fusion_train_previous(app_train, previous_application):
    """Fusionne le train avec previous_application selon SK_ID_CURR."""
    return app_train.merge(previous_application, on='SK_ID_CURR', how='inner')


def concat_train_test(app_train, app_test):
    """Concatène train et test en un seul dataframe."""
    data = pd.concat([app_train, app_test], axis=0, ignore_index=True)
    print('Train:', app_train.shape)
    print('Test:', app_test.shape)
    print('>>> Data:', data.shape)
    return data


def ajouter_compte_pret_precedent(data, bureau):
    """Ajoute le nombre de prêts précédents par client."""
    previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns={'SK_ID_BUREAU': 'PREVIOUS_LOANS_COUNT'})
    return data.merge(previous_loan_counts, on='SK_ID_CURR', how='left')


def ajouter_moyenne_balance_bureau(bureau_balance, bureau):
    """Calcule la moyenne des balances mensuelles par SK_ID_BUREAU et fusionne avec bureau."""
    numeric_cols = bureau_balance.select_dtypes(include='number').columns.tolist()
    if 'SK_ID_BUREAU' in numeric_cols:
        numeric_cols.remove('SK_ID_BUREAU')
    bureau_bal_mean = bureau_balance.groupby('SK_ID_BUREAU', as_index=False)[numeric_cols].mean()
    bureau_bal_mean = bureau_bal_mean.rename(columns={'MONTHS_BALANCE': 'MONTHS_BALANCE_MEAN'})
    bureau_full = bureau.merge(bureau_bal_mean, on='SK_ID_BUREAU', how='left').drop('SK_ID_BUREAU', axis=1)
    return bureau_full


def ajouter_moyenne_par_client(bureau_full, data):
    """Calcule la moyenne des colonnes numériques dans bureau_full par client et fusionne avec data."""
    bureau_mean = bureau_full.groupby('SK_ID_CURR', as_index=False)[
        bureau_full.select_dtypes(include='number').columns].mean().add_prefix('PREV_BUR_MEAN_')
    bureau_mean.rename(columns={'PREV_BUR_MEAN_SK_ID_CURR': 'SK_ID_CURR'}, inplace=True)
    return data.merge(bureau_mean, on='SK_ID_CURR', how='left')


def ajouter_nb_previous_app(data, previous_application):
    """Ajoute le nombre de demandes de crédit précédentes."""
    previous_counts = previous_application.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count()
    previous_counts.rename(columns={'SK_ID_PREV': 'PREVIOUS_APPLICATION_COUNT'}, inplace=True)
    return data.merge(previous_counts, on='SK_ID_CURR', how='left')


def enrichir_previous_application(previous_application, credit_card_balance, installments_payments, POS_CASH_balance):
    """Ajoute les moyennes de plusieurs sources à previous_application via SK_ID_PREV."""

    # CREDIT CARD BALANCE
    numeric_cols = credit_card_balance.select_dtypes(include=['number']).columns
    if 'SK_ID_PREV' not in numeric_cols:
        numeric_cols = numeric_cols.tolist() + ['SK_ID_PREV']
    credit_card_balance_mean = credit_card_balance[numeric_cols].groupby('SK_ID_PREV', as_index=False).mean()
    credit_card_balance_mean = credit_card_balance_mean.rename(
        columns={col: f'CARD_MEAN_{col}' for col in credit_card_balance_mean.columns if col != 'SK_ID_PREV'}
    )
    previous_application = previous_application.merge(credit_card_balance_mean, on='SK_ID_PREV', how='left')

    # INSTALLMENTS PAYMENTS
    install_mean = installments_payments.groupby('SK_ID_PREV', as_index=False).mean().add_prefix('INSTALL_MEAN_')
    install_mean.rename(columns={'INSTALL_MEAN_SK_ID_PREV': 'SK_ID_PREV'}, inplace=True)
    previous_application = previous_application.merge(install_mean, on='SK_ID_PREV', how='left')

    # POS CASH BALANCE
    POS_mean = POS_CASH_balance.groupby('SK_ID_PREV', as_index=False).mean(numeric_only=True).add_prefix('POS_MEAN_')
    POS_mean.rename(columns={'POS_MEAN_SK_ID_PREV': 'SK_ID_PREV'}, inplace=True)
    previous_application = previous_application.merge(POS_mean, on='SK_ID_PREV', how='left')

    return previous_application


def moyenne_previous_application_par_client(previous_application, data):
    """Ajoute la moyenne des colonnes numériques de previous_application par client dans data."""
    previous_application_numeric = previous_application.select_dtypes(include=['number'])
    prev_appl_mean = previous_application_numeric.groupby('SK_ID_CURR', as_index=False).mean()
    prev_appl_mean.columns = ['SK_ID_CURR'] + [f'PREV_APPL_MEAN_{col}' for col in prev_appl_mean.columns if col != 'SK_ID_CURR']
    return data.merge(prev_appl_mean, on='SK_ID_CURR', how='left')