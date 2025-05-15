def save_target_column(data_train: pd.DataFrame, output_path: str = "TARGET.csv") -> None:
    """
    Extrait et sauvegarde la colonne TARGET de l'ensemble d'entraînement.

    Args:
        data_train (pd.DataFrame): Dataset d'entraînement incluant la colonne TARGET.
        output_path (str): Chemin du fichier de sortie pour la colonne TARGET.
    """
    target = data_train["TARGET"]
    target.to_csv(output_path, index=True)
    print(f"Colonne TARGET sauvegardée sous : {output_path}")
    
    
def save_dataframe_zip(df, filename, csv_name): 
    compression_opts = dict(method='zip', archive_name=csv_name)
    df.to_csv(filename, index=True, compression=compression_opts)