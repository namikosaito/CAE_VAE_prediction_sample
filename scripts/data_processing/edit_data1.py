import pandas as pd
import os

data_path = "../../data/sample_img_data"

dirs = os.listdir(data_path)
dirs.sort()

for dir in dirs:
    dir_path = os.path.join(data_path, dir)  # ディレクトリのパスを作成
    files = os.listdir(dir_path)
    files.sort()

    for file in files:
        if file.endswith('data.csv'):  # CSVファイルのみを操作
            file_path = os.path.join(dir_path, file)  # CSVファイルのパスを作成
            # CSVファイルを読み込む
            df = pd.read_csv(file_path, header=None)  # 列名がないことを示すために header=None を指定

            # 1行目を削除
            df = df.iloc[1:]

            # 'f'以降の列を削除
            df = df.iloc[:, :5]

            # 5行目をすべて3に書き換え
            df.iloc[:,4] = 3

            # 1列目の重複した行を削除して上詰め
            df = df.drop_duplicates(subset=df.columns[0], keep='first')

            # 1retumeを削除
            df = df.iloc[:,0]

            # 新しいCSVファイルとして保存（ファイル名にディレクトリ名を含める）
            new_file_name = os.path.splitext(file)[0] + '_1.csv'
            new_file_path = os.path.join(dir_path, new_file_name)
            df.to_csv(new_file_path, index=False, header=False)  # 列名も保存しないように header=False を指定
