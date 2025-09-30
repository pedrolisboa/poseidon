import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from poseidon.dataset.loader import SpectrogramLoader
import os 

class SonarCrossValidator:
    """
    Handles the creation of stratified cross-validation splits at the run level,
    with support for stratification across multiple metadata columns.
    """
    def __init__(self, metadata_df, target_column, stratify_columns, 
                 n_splits=5, random_state=42):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing all run metadata.
            target_column (str): The primary column for ground truth labels (e.g., 'Ship Size').
            stratify_columns (list): A list of columns to use for stratification.
                                     Should include the target_column.
            n_splits (int): The number of folds for cross-validation.
            random_state (int): Seed for reproducibility.
        """
        self.df = metadata_df.copy()
        self.target_column = target_column
        self.stratify_columns = stratify_columns
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.splits = self._create_splits()

        labels = metadata_df[target_column]       
        classes = np.unique(labels)
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        print("Created class mapping:", self.class_to_idx)

    def _create_splits(self):
        """
        Generates the stratified splits. For multi-column stratification, it
        creates a composite key from the specified columns.
        """
        print(f"Creating {self.n_splits} stratified splits based on columns: {self.stratify_columns}")
        
        # --- The Multi-Column Stratification Trick ---
        # Combine multiple columns into a single string column to stratify on.
        stratify_key = self.df[self.stratify_columns].apply(
            lambda row: '_'.join(row.values.astype(str)),
            axis=1
        )
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # We store the DataFrame indices for each split
        # We use .index to be robust to any prior filtering of the DataFrame
        indices = self.df.index.to_numpy()
        return list(skf.split(indices, stratify_key))

    def get_fold_data(self, fold_idx, processed_data_root):
        """
        Retrieves the data for a specific fold.
        MODIFIED: Now uses SpectrogramLoader.

        Args:
            fold_idx (int): The index of the fold to retrieve.
            processed_data_root (str): The ROOT PATH to the cached spectrograms.
        """
        # ... (getting train_df and test_df is the same) ...
        train_indices, test_indices = self.splits[fold_idx]
        train_df = self.df.loc[train_indices]
        test_df = self.df.loc[test_indices]
        
        def _collect_data(subset_df):
            data_list = []
            for _, row in subset_df.iterrows():
                class_name = str(row[self.target_column])
                dataset_folder = str(row['Dataset'])
                run_id = str(row['ID'])

                # TODO: pass the int casting to early steps in the processing chain
                run_name = f"{dataset_folder}-{int(run_id):04d}"
                label_int = self.class_to_idx[int(class_name)]
                
                # Construct the filepath directly
                filepath = os.path.join(processed_data_root, class_name, f"{run_name}.npz")

                if os.path.exists(filepath):
                    # --- CRITICAL CHANGE ---
                    # Create a stateless loader instead of a caching record.
                    loader = SpectrogramLoader(filepath)
                    data_list.append((loader, label_int))
                else:
                    print(f"Warning: File '{filepath}' not found.")
            return data_list
            
        train_data = _collect_data(train_df)
        test_data = _collect_data(test_df)
        
        return train_data, test_data


    def calculate_fold_weights(self, fold_idx):
        """
        Calculates class weights for the training set of a specific fold to handle imbalance.
        The formula used is: n_samples / (n_classes * count_of_each_class).

        Args:
            fold_idx (int): The index of the fold for which to calculate weights.

        Returns:
            np.ndarray: An array of weights corresponding to each class, ordered by class index.
        """
        train_indices, _ = self.splits[fold_idx]
        train_df = self.df.iloc[train_indices]

        # Get the labels for the training set of the specified fold
        train_labels = train_df[self.target_column]

        # Calculate class counts
        class_counts = train_labels.value_counts()
        
        # Total number of samples in the training set
        n_samples = len(train_labels)
        
        # Number of unique classes
        n_classes = len(self.class_to_idx)

        # Initialize a weights array
        weights = np.zeros(n_classes)

        # Calculate weight for each class and place it in the correct index position
        for class_label, count in class_counts.items():
            class_idx = self.class_to_idx[class_label]
            weights[class_idx] = n_samples / (n_classes * count)
            
        return weights