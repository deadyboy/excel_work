import pandas as pd

# 1. 定义输入输出路径（修改点）
file1 = '/data1/jianf/Excel work/processed_patient_data_with_units.csv'  # 您的第一个 CSV 文件
file2 = '/data1/jianf/Merged_Patients_Output/1_主要变量表_Sheet02.csv'  # 您的第二个 CSV 文件
output_file = '/data1/jianf/Excel work/final_combined.xlsx'  # 输出必须是 xlsx

# 2. 读取数据 (处理 CSV 建议加上 encoding 参数)
# 如果读取报错，请尝试 encoding='gbk'
df1 = pd.read_csv(file1, encoding='utf-8')
df2 = pd.read_csv(file2, encoding='utf-8')

# 3. 写入 Excel 的不同 Sheet
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Data_Source_A', index=False)
    df2.to_excel(writer, sheet_name='Data_Source_B', index=False)

print(f"处理完成：已将 {file1} 和 {file2} 合并至 {output_file}")