import pandas as pd
import datetime

def to_csv_with_comment(df:pd.DataFrame, comment:str,output_filename:str):
    with open(output_filename, 'a') as f:
        # 現在の日時を取得
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # コメント行を追加
        f.write(f'# File created on: {current_time}\n')
        f.write(comment)
        df.to_csv(f,index=False)
