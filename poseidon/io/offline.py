"""
    Projeto Marinha do Brasil

    Autor: Pedro Henrique Braga Lisboa (pedro.lisboa@lps.ufrj.br)
    Laboratorio de Processamento de Sinais - UFRJ
    Laboratorio de de Tecnologia Sonar - UFRJ/Marinha do Brasil
"""
from __future__ import print_function, division

import os
import warnings
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

def load_raw_data(input_db_path, verbose=0):
    """
        Loads sonar audio datafiles on memory. 

        This function returns a nested hashmap associating each run audio data with its
        class and filename. The audio information is composed by 
        the frames stored in a numpy array and the file informed sample rate.
        
        E.g. for database '4classes' the returned dictionary will be set like:
        
        ClassA:
            navio10.wav: 
                signal: np.array
                sample_rate: np.float64
            navio11.wav: 
                signal: np.array
                sample_rate: np.float64
        ClassB:
            navio20.wav: 
                ...
            navio21.wav:
                ...
            ...
        ...
            
        params:
            input_data_path (string): 
                path to database folder
        return (SonarDict): 
                nested dicionary in which the basic unit contains
                a record of the audio (signal key) in np.array format
                and the sample_rate (fs key) stored in floating point. 
                The returned object also contains a method for applying
                functions over the runs (see SonarDict.apply).
                the map is made associating each tuple to the corresponding
                name of the run (e.g. )
    """

    if verbose:
        print('Reading Raw data in path %s' % input_db_path)

    class_folders = [folder for folder in os.listdir(input_db_path)
                        if not folder.startswith('.')]
    raw_data = dict()
    
    for cls_folder in class_folders:
        runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
        if not runfiles:  # No files found inside the class folder
            if verbose:
                print('Empty directory %s' % cls_folder)
            continue
        if verbose:
            print('Reading %s' % cls_folder)

        runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
        runpaths = [os.path.join(input_db_path, cls_folder, runfile)
                    for runfile in runfiles]
        audio_data = [read_audio_file(runpath) for runpath in runpaths]

        raw_data[cls_folder] = {
            runfile: RunRecord(signal, fs)
            for runfile, (signal, fs)  in zip(runfiles, audio_data)
        }

    return SonarDict(raw_data)

class RunRecord(dict):
    """
    Basic dicionary for storing the runs
    binding the data with its respective metadata(sample rate)
    This wrapper was made to standardize the keynames. 
    """

    def __init__(self, signal, fs):
        self.__dict__['signal'] = signal
        self.__dict__['fs'] = fs

class SonarDict(dict):
    """ 
    Wrapper for easy application of preprocessing functions 
    """
    def __init__(self, raw_data):
        super(SonarDict, self).__init__(raw_data)

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
        print([a for a, b in cls_data.items()])
        return {
            run_name: fn(raw_data, *args, **kwargs)
            for run_name, raw_data in cls_data.items()
        }

def read_audio_file(filepath):
    signal, fs = sf.read(filepath)

    return signal, fs

