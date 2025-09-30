"""
    Projeto Marinha do Brasil

    Autor: Pedro Henrique Braga Lisboa (pedro.lisboa@lps.ufrj.br)
    Laboratorio de Processamento de Sinais - UFRJ
    Laboratorio de de Tecnologia Sonar - UFRJ/Marinha do Brasil
"""
from __future__ import print_function, division

import os
import h5py
import warnings
import numpy as np
import pandas as pd # You will need to install pandas: pip install pandas
import soundfile as sf
import enum
import gc 

import concurrent.futures
from tqdm import tqdm

# TODO: place IARA credits here
class Target(enum.Enum):
    # https://www.mdpi.com/2072-4292/11/3/353
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    BACKGROUND = 3

    @staticmethod
    def classify(ship_length: float) -> 'Target':
        if np.isnan(ship_length):
            return 3

        if ship_length < 50:
            return 0
        if ship_length < 100:
            return 1
        return 2

    @staticmethod
    def classify_value(value: pd.DataFrame) -> float:
        try:
            return Target.classify(value)
        except ValueError:
            return np.nan

# This new helper function will be executed by each thread
def _process_single_file(fn, run_name, run_data, output_filepath, *args, **kwargs):
    """
    Worker function for a single thread. It processes one audio file,
    saves the result, and ensures memory is released.
    """
    try:
        # This triggers the LazyRunRecord to load the raw audio
        processed_result = fn(run_data, *args, **kwargs)
        
        # Save the result
        np.savez_compressed(output_filepath, **processed_result)
        
        # Return the name of the run on success
        return run_name
    except Exception as e:
        # Return the exception if something goes wrong
        return e
    finally:
        # CRITICAL: Always release the memory for this run
        if hasattr(run_data, 'release'):
            run_data.release()

class LazySpectrogramRecord(object):
    """
    A descriptor for a single processed run (spectrogram) that loads data
    from a cached .npz file only when an attribute is accessed.
    
    This allows you to hold references to an entire processed dataset in memory
    without loading the actual (potentially large) spectrogram arrays.
    """
    def __init__(self, filepath):
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be a string, but got {type(filepath)}")
        self.filepath = filepath
        self._data = None  # This will cache the loaded data

    def _load_data(self):
        """Private method to load data from the .npz file. Called only once."""
        if self._data is None:
            try:
                # np.load returns an NpzFile object, which is a dict-like object
                # that loads arrays lazily from the zip container.
                self._data = np.load(self.filepath)
            except Exception as e:
                print(f"Error loading processed file: {self.filepath}")
                raise e

    def __getitem__(self, key):
        """
        Provides dictionary-like access (e.g., record['sxx']).
        Loads data from disk on the first access to any key.
        """
        self._load_data()
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in {self.filepath}. Available keys: {list(self._data.keys())}")

    def keys(self):
        """Returns the available keys in the .npz file (e.g., ['sxx', 'freq', 'time'])."""
        self._load_data()
        return self._data.keys()

    def close(self):
        """Closes the underlying NpzFile object to release the file handle."""
        if self._data is not None:
            self._data.close()

    def __repr__(self):
        """Provides a helpful representation of the object."""
        status = "Loaded" if self._data is not None else "Not Loaded (on disk)"
        return f"<LazySpectrogramRecord(file='{os.path.basename(self.filepath)}', status='{status}')>"


class LazyRunRecord(object):
    """
    A descriptor for a single run that loads audio data from disk only when
    the 'signal' or 'fs' attributes are accessed. This is essential for
    handling datasets that are too large to fit into memory.
    """
    def __init__(self, filepath):
        if not isinstance(filepath, str):
            raise TypeError(f"filepath must be a string, but got {type(filepath)}")
        self.filepath = filepath
        self._signal = None
        self._fs = None

    def _load_audio(self):
        """Private method to load audio data. It's called only once."""
        if self._signal is None or self._fs is None:
            try:
                self._signal, self._fs = read_audio_file(self.filepath)
            except Exception as e:
                print(f"Error loading audio file: {self.filepath}")
                raise e

    def __getitem__(self, key):
        """
        Behaves like a dictionary to maintain compatibility with SonarDict.apply.
        """
        if key not in ['signal', 'fs']:
            raise KeyError(f"Key '{key}' not supported. Available keys are 'signal' and 'fs'.")
        
        # Load audio from disk on first access
        self._load_audio()
        
        if key == 'signal':
            return self._signal
        if key == 'fs':
            return self._fs

    def __repr__(self):
        """Provides a helpful representation of the object."""
        status = "Loaded" if self._signal is not None else "Not Loaded"
        return f"<LazyRunRecord(file='{os.path.basename(self.filepath)}', status='{status}')>"

    def release(self):
        """
        Explicitly releases the loaded signal and fs from memory by removing
        the reference to them.
        """
        if self._signal is not None:
            #print(f"Releasing memory for {os.path.basename(self.filepath)}")
            self._signal = None
            self._fs = None
            # Optionally, call the garbage collector to encourage it to clean up.
            gc.collect()


def load_sonar_from_csv(csv_path, data_root_path, target_column, verbose=1):
    """
    Loads sonar dataset structure from a CSV file for large datasets.

    This function reads a metadata CSV, groups runs by the specified target_column,
    and creates a SonarDict populated with LazyRunRecord objects. The audio data
    itself is NOT loaded into memory until it is explicitly accessed.

    File paths are constructed as: data_root_path/Dataset/Dataset-ID.wav

    Args:
        csv_path (str):
            Path to the metadata .csv file.
        data_root_path (str):
            The root directory containing the dataset folders (e.g., '/path/to/my/data').
        target_column (str):
            The name of the column in the CSV to use as the class label for grouping.
            E.g., "Sea state", "Ship Size".
        verbose (int):
            Verbosity level.

    Returns:
        (SonarDict):
            A nested dictionary-like object where keys are the unique values from the
            target_column. Each value is another dictionary mapping a run_name to a
            LazyRunRecord object.
    """
    if verbose:
        print(f"Reading dataset metadata from: {csv_path}")
        print(f"Using data root path: {data_root_path}")
        print(f"Grouping by target column: '{target_column}'")

    try:
        df = pd.read_csv(csv_path)
        df['Length'] = df['Length'].apply(lambda x: np.nan if x == ' - ' else float(x))
        df['Ship Length Class'] = df['Length'].apply(Target.classify_value)

    except FileNotFoundError:
        print(f"Error: Metadata CSV file not found at {csv_path}")
        return SonarDict({})
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the CSV file.")
        print(f"Available columns are: {list(df.columns)}")
        return SonarDict({})


    raw_data = {}
    
    # Group the dataframe by the desired target class
    for class_label, group in df.groupby(target_column):
        
        class_name = str(class_label) # Ensure dictionary key is a string
        if verbose > 1:
            print(f"Processing class: {class_name}")

        raw_data[class_name] = {}
        # Iterate over each run (row) in the class group
        for _, row in group.iterrows():
            dataset_folder = str(row['Dataset'])
            run_id = int(row['ID'])
            
            # Construct a unique name and the full path for the audio file
            run_name = f"{dataset_folder}-{run_id:04d}"
            wav_filename = f"{run_name}.wav"
            filepath = os.path.join(data_root_path, dataset_folder, wav_filename)

            if os.path.exists(filepath):
                # Instead of loading the data, create a lazy loader object
                raw_data[class_name][run_name] = LazyRunRecord(filepath)
            else:
                warnings.warn(f"Audio file not found and will be skipped: {filepath}")

    if verbose:
        print(f"Dataset structure loaded. Found {len(raw_data)} classes.")
        for class_name, runs in raw_data.items():
            print(f"  - Class '{class_name}': {len(runs)} runs")

    return SonarDict(raw_data)


class SonarDict(dict):
    """ 
    Wrapper for easy application of preprocessing functions 
    """
    def __init__(self, raw_data):
        super(SonarDict, self).__init__(raw_data)

    @staticmethod
    def from_hdf5(filepath):
        f = h5py.File(filepath, 'r')
        raw_data = SonarDict.__level_from_hdf5(f)
        f.close()
        return SonarDict(raw_data)
        
    @staticmethod
    def __level_from_hdf5(group_level):
        level_dict = dict()
        for key in group_level.keys():
            if isinstance(group_level[key], h5py._hl.group.Group):
                level_dict[key] = SonarDict.__level_from_hdf5(group_level[key])
            elif isinstance(group_level[key], h5py._hl.dataset.Dataset):
                # if isinstance(group_level[key].dtype, 'float64')
                level_dict[key] = group_level[key][()]
            else:
                raise ValueError

        return level_dict


    def to_hdf5(self, filepath):
        f = h5py.File(filepath, 'w')
        SonarDict.__level_to_hdf5(self, f, '')
        f.close()

    @staticmethod
    def __level_to_hdf5(dictionary_level, f, dpath):
        for key in dictionary_level.keys():
            ndpath = dpath + '/%s' % key
            if isinstance(dictionary_level[key], dict):
                SonarDict.__level_to_hdf5(dictionary_level[key], f, ndpath)
            else:
                if isinstance(dictionary_level[key], np.ndarray):
                    dtype = dictionary_level[key].dtype
                else:
                    dtype = type(dictionary_level[key])
                f.create_dataset(ndpath, data=dictionary_level[key], dtype=dtype)

    def apply(self, fn,*args, **kwargs):
        """ 
        Apply a function over each run of the dataset.

        params:
            fn: callable to be applied over the data. Receives at least
                one parameter: dictionary (RunRecord)
            args: optional params to fn
            kwargs: optional named params to fn
        
        return:
            new SonarDict object with the processed data. The inner structure
            of signal, sample_rate pair is mantained, which allows for chaining
            several preprocessing steps.

        """
        sonar_cp = self.copy()

        return SonarDict({
            cls_name: self._apply_on_class(cls_data, fn, *args, **kwargs) 
            for cls_name, cls_data in sonar_cp.items()
        })
        

    def _apply_on_class(self, cls_data, fn, *args, **kwargs):
        """
        Apply a function over each run signal of a single class.
        Auxiliary function for applying over the dataset
        """
        return {
            run_name: fn(raw_data, *args, **kwargs)
            for run_name, raw_data in cls_data.items()
        }
        

    def process_and_cache(self, fn, cache_path, max_workers=8, *args, **kwargs):
        """
        A parallelized version of process_and_cache using ThreadPoolExecutor.

        Args:
            fn (callable):
                The processing function to apply (e.g., lofar).
            cache_path (str):
                The root directory where processed files will be saved.
            max_workers (int):
                The number of concurrent threads to use. This is the primary
                control for peak memory usage.
        """
        print(f"Starting PARALLEL processing with {max_workers} workers.")
        
        # --- 1. GATHER TASKS ---
        # First, identify all files that need to be processed.
        tasks = []
        skipped_count = 0
        for class_name, class_data in self.items():
            class_cache_path = os.path.join(cache_path, str(class_name))
            os.makedirs(class_cache_path, exist_ok=True)

            for run_name, run_data in class_data.items():
                output_filepath = os.path.join(class_cache_path, f"{run_name}.npz")
                if os.path.exists(output_filepath):
                    skipped_count += 1
                else:
                    # Add all necessary info for the worker to the tasks list
                    tasks.append((run_name, run_data, output_filepath))

        if not tasks:
            print("All files are already processed and cached.")
            print(f"Skipped (already cached): {skipped_count} files.")
            return

        print(f"Found {len(tasks)} files to process. {skipped_count} files are already cached.")

        # --- 2. EXECUTE IN PARALLEL ---
        processed_count = 0
        failed_count = 0
        
        # The ThreadPoolExecutor manages the pool of worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            # Submit all tasks to the executor. The worker function is _process_single_file.
            future_to_run_name = {
                executor.submit(_process_single_file, fn, run_name, run_data, output_filepath, *args, **kwargs): run_name
                for run_name, run_data, output_filepath in tasks
            }

            # Use tqdm to create a progress bar as tasks complete
            for future in tqdm(concurrent.futures.as_completed(future_to_run_name), total=len(tasks), desc="Processing runs"):
                run_name = future_to_run_name[future]
                try:
                    # .result() will return the value from the worker function
                    # or re-raise any exception that occurred.
                    result = future.result()
                    if isinstance(result, Exception):
                        print(f"\nERROR processing {run_name}: {result}")
                        failed_count += 1
                    else:
                        processed_count += 1
                except Exception as e:
                    print(f"\nCRITICAL ERROR for run {run_name}: {e}")
                    failed_count += 1

        print("\n--- Caching Summary ---")
        print(f"Successfully processed and saved: {processed_count} new files.")
        print(f"Skipped (already cached):      {skipped_count} files.")
        print(f"Failed to process:             {failed_count} files.")

        
def read_audio_file(filepath):
    signal, fs = sf.read(filepath)

    return signal, fs

def load_processed_data(cache_path, verbose=1):
    """
    Scans a cache directory created by `process_and_cache` and returns a
    SonarDict populated with LazySpectrogramRecord objects.

    This allows lazy loading of the processed spectrogram data for analysis.

    Args:
        cache_path (str):
            The root directory where the processed .npz files are stored.
        verbose (int):
            Verbosity level.

    Returns:
        (SonarDict):
            A nested dictionary-like object structured just like the original,
            but pointing to the processed data on disk.
    """
    if not os.path.isdir(cache_path):
        print(f"Error: Cache directory not found at {cache_path}")
        return SonarDict({})

    if verbose:
        print(f"Loading processed data structure from: {cache_path}")

    processed_data = {}
    
    class_names = [d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]

    for class_name in class_names:
        processed_data[class_name] = {}
        class_dir = os.path.join(cache_path, class_name)
        
        for filename in os.listdir(class_dir):
            if filename.endswith('.npz'):
                run_name = filename.replace('.npz', '')
                filepath = os.path.join(class_dir, filename)
                # Create a lazy record for the processed file
                processed_data[class_name][run_name] = LazySpectrogramRecord(filepath)
    
    if verbose:
        print(f"Processed dataset structure loaded. Found {len(processed_data)} classes.")
        for class_name, runs in processed_data.items():
            print(f"  - Class '{class_name}': {len(runs)} runs")

    return SonarDict(processed_data)