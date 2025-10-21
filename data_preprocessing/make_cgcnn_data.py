import pandas as pd
import shutil
import os

csv_file = 'mof_train_77K_100bar_langmuir.csv'
df = pd.read_csv(csv_file)

os.makedirs('cif', exist_ok=True)
with open('id_prop.csv', 'w') as f:
    for _, row in df.iterrows():
        cif_name = os.path.basename(row['cif_path'])
        # 复制cif到cif目录
        try:
            shutil.copy(row['cif_path'], f'cif/{cif_name}')
        except Exception as e:
            print(f"缺少cif文件: {row['cif_path']}, 跳过")
            continue
        f.write(f"{cif_name},{row['target']}\n")
print("已生成id_prop.csv和cif目录。")
