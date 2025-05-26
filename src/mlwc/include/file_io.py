"""
Module relating I/O
"""

import datetime

import pandas as pd


def to_csv_with_comment(df: pd.DataFrame, comment: str, output_filename: str) -> None:
    """pandas to_csv with additional comments"""
    with open(output_filename, "a", encoding="utf-8") as f:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"# File created on: {current_time}\n")
        f.write(comment)
        df.to_csv(f, index=False)
