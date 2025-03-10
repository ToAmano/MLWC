import pandas as pd
import numpy as np
import argparse
import os
import datetime
import yaml
from pathlib import Path
import __version__
from include.mlwc_logger import setup_library_logger
logger = setup_library_logger("MLWC."+__name__)


class average_diel:
    def __init__(self,list_input_filename:list[str],window:int=1,max_freq_kayser:float=4000):
        """
        Initializes the average_diel class.

        Args:
            list_input_filename (list[str]): A list of input filenames.
            window (int, optional): The window size for the moving average. Defaults to 1.
            max_freq_kayser (int, optional): The maximum frequency in Kayser units. Defaults to 4000.
        """
        self.list_input_filename:list[str] = list_input_filename
        self.window:int = window
        self.max_freq_kayser:float = max_freq_kayser
    
    def read_file(self):
        """
        Reads multiple CSV files into a list of pandas DataFrames.

        The method reads each file specified in `self.list_input_filename`, checks for its existence,
        and ensures that the 'freq_kayser' column is present. Unnamed columns are dropped, and
        'freq_kayser' is set as the index.

        Returns:
            list[pd.DataFrame]: A list of pandas DataFrames, each representing the data from an input file.

        Raises:
            FileNotFoundError: If any of the specified input files does not exist.
            ValueError: If the 'freq_kayser' column is not found in any of the input files.
        """
        list_df:list[pd.DataFrame] = []
        for input_filename in self.list_input_filename:
            logger.info(f"Reading {input_filename}")
            if not os.path.exists(input_filename):
                raise FileNotFoundError(f"{input_filename} not found")
            df_tmp = pd.read_csv(input_filename,comment="#")
            if "freq_kayser" not in df_tmp.columns:
                raise ValueError("freq_kayser is not in columns")
            for col in df_tmp.columns:
                if col.startswith("Unnamed:"):
                    df_tmp = df_tmp.drop(col,axis=1)
            df_tmp = df_tmp.set_index("freq_kayser")
            list_df.append(df_tmp)
        return list_df
        
    @classmethod
    def average_diel(cls,df:pd.DataFrame,window:int=1)->pd.DataFrame:
        """
        Averages dielectric data over multiple DataFrames and applies a moving average.

        This method takes a DataFrame, concatenates the data (if it is a list), averages the data for each frequency,
        and then applies a moving average filter to smooth the data.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing dielectric data.
            window (int, optional): The window size for the moving average. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the averaged and smoothed dielectric data.
        """
        # concat dataframes
        df = pd.concat(df.copy())
        # average over the same frequency
        df_mean = df.groupby(df.index).mean()
        # moving average
        window_func = np.ones(window)/window
        for col in df_mean.columns:
            if col == "freq_kayser":
                continue
            elif col == "freq_thz":
                continue
            else:
                df_mean[col] = np.convolve(df_mean[col], window_func, mode="same")
        df_mean = df_mean.reset_index()
        return df_mean
    
    @classmethod
    def truncate_diel(cls,df:pd.DataFrame,max_freq_kayser:int=4000):
        """
        Truncates the DataFrame to a maximum frequency in Kayser units.

        This method filters the DataFrame to include only rows where the 'freq_kayser'
        column is less than or equal to the specified `max_freq_kayser`.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing dielectric data with a 'freq_kayser' column.
            max_freq_kayser (int, optional): The maximum frequency in Kayser units. Defaults to 4000.

        Returns:
            pd.DataFrame: A DataFrame containing the truncated dielectric data.
        """
        df = df.copy()
        df = df[df["freq_kayser"] <= max_freq_kayser]
        return df
    
    def save_file(self, df:pd.DataFrame)-> None:
        """
        Saves the processed DataFrame to a CSV file.

        This method saves the provided DataFrame to a CSV file with a filename derived from the input filename
        and the averaging window size. It includes a header with metadata such as the creation date,
        the CPextract version, and the parameters used for processing.

        Args:
            df (pd.DataFrame): The DataFrame to save.

        Raises:
            FileExistsError: If the output file already exists.
        """
        output_filename:str = self.list_input_filename[0]+f"_average_{self.window}.csv"
        if os.path.exists(output_filename):
            raise FileExistsError(f"{output_filename} already exists")
        # df.to_csv(self.output_file, index=False)
        with open(output_filename, 'a') as f:
            # 現在の日時を取得
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # コメント行を追加
            f.write(f'# File created on: {current_time}\n')
            f.write(f'# File generated by CPextract.py diel average version {__version__.__version__}.\n')
            f.write(f'# Parameters: window={self.window}, filename={self.list_input_filename}\n')
            f.write('# Data below:\n')
            df.to_csv(f,index=False)
        print(f"Results saved to {output_filename}")
        
        
    def execute(self):
        """
        Executes the entire sequence of operations: reading, averaging, truncating, and saving.

        This method orchestrates the process of reading the input files, averaging the dielectric data,
        truncating the data to the specified maximum frequency, and saving the processed data to a CSV file.
        """
        df:pd.DataFrame = self.read_file()
        df:pd.DataFrame = self.average_diel(df,self.window)
        df:pd.DataFrame = self.truncate_diel(df,self.max_freq_kayser)
        self.save_file(df)

def check_filename(list_filename:list[str]):
    """
    Checks if a list of filenames exist.

    This function takes a list of filenames, checks for the existence of each file,
    and returns a list of valid filenames.

    Args:
        list_filename (list[str]): A list of filenames to check.

    Returns:
        list[str]: A list of valid filenames that exist.

    Raises:
        FileNotFoundError: If any of the specified files does not exist.
    """
    list_filename_out:list[str] = []
    for filename in list_filename:
        filename = filename.strip("")
        logger.info(f"Checking {filename}")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found")
        list_filename_out.append(filename)
    return list_filename_out

def command_diel_average(args:argparse.Namespace):
    """
    Command-line interface function to execute dielectric averaging.

    This function takes command-line arguments, reads a YAML file containing a list of filenames,
    checks the existence of those files, initializes the `average_diel` class, and executes the averaging process.

    Args:
        args (argparse.Namespace): An object containing the command-line arguments.

    Returns:
        int: 0 upon successful execution.
    """
    with open(args.Filename) as file:
        yml = yaml.safe_load(file)
        filelist = check_filename(yml["filename"])
    processor = average_diel(filelist, int(args.window), int(args.maxfreq))
    processor.execute()
    return 0