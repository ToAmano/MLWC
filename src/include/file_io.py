import pandas as pd
import datetime


def to_csv_with_comment(df: pd.DataFrame, comment: str, output_filename: str):
    """pandas to_csv with additional comments"""
    with open(output_filename, 'a') as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f'# File created on: {current_time}\n')
        f.write(comment)
        df.to_csv(f, index=False)
