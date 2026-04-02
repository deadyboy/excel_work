import pandas as pd
import json

# 1. 读取数据
df = pd.read_excel('/data1/jianf/Excel work/ecmo-0015301694.xlsx', header=1)

# 【防御机制 1】向下填充患者ID，防止平铺错位导致的底部数据丢失患者归属
df['患者ID'] = df['患者ID'].ffill()

# 2. 提取并去重患者基础信息
basic_cols = [
    '患者ID', 
    '【患者基本信息.病人姓名】', 
    '【患者基本信息.病人性别】', 
    '【住院登记.病人年龄】',
    '【住院登记.入院时间】',
    '【住院登记.出院时间】'
]
df_basic = df[basic_cols].dropna(subset=['患者ID']).drop_duplicates(subset=['患者ID'], keep='first')

# 【防御机制 2】主键正则清洗函数
def clean_id(series):
    """强制将所有主键转换为绝对干净的字符串，剔除浮点转换带来的 .0 以及多余空格"""
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

# 3. 提取元数据与明细，并实施主键清洗
df_report_meta = df[['【检验报告.检验报告号】', '【检验报告.检验大类名称】', '【检验报告.报告时间】']].dropna(subset=['【检验报告.检验报告号】']).copy()
df_report_meta['【检验报告.检验报告号】'] = clean_id(df_report_meta['【检验报告.检验报告号】'])
df_report_meta = df_report_meta.drop_duplicates(subset=['【检验报告.检验报告号】'], keep='first')

# 【新增逻辑】：在明细表中额外提取 '【检验结果.检验项目结果单位】'
df_report_items = df[[
    '患者ID', 
    '【检验结果.检验报告号】', 
    '【检验结果.检验项目中文名称】', 
    '【检验结果.检验项目结果】',
    '【检验结果.检验项目结果单位】'
]].dropna(subset=['【检验结果.检验报告号】', '【检验结果.检验项目中文名称】']).copy()

df_report_items['【检验结果.检验报告号】'] = clean_id(df_report_items['【检验结果.检验报告号】'])

# 安全执行内连接
df_merged = pd.merge(
    df_report_items,
    df_report_meta,
    left_on='【检验结果.检验报告号】',
    right_on='【检验报告.检验报告号】',
    how='inner'
)

if df_merged.empty:
    raise ValueError("严重异常：内连接后数据量为 0！")

# 4 & 5. 带有单位拼接逻辑的序列化函数
def build_patient_category_series(group):
    reports = []
    for report_id, sub_group in group.groupby('【检验结果.检验报告号】'):
        time_val = sub_group['【检验报告.报告时间】'].iloc[0]
        report_dict = {"时间": str(time_val)}
        
        for _, row in sub_group.iterrows():
            item_name = str(row['【检验结果.检验项目中文名称】']).strip()
            item_val = str(row['【检验结果.检验项目结果】']).strip()
            item_unit = str(row['【检验结果.检验项目结果单位】']).strip()
            
            # 过滤无效的项目名称
            if item_name and item_name.lower() != 'nan':
                # 【新增逻辑】：如果单位存在且不为空/nan，则将其与结果拼接
                if item_unit and item_unit.lower() not in ['nan', 'none', '']:
                    final_val = f"{item_val} {item_unit}"
                else:
                    final_val = item_val
                    
                report_dict[item_name] = final_val
                
        reports.append(report_dict)
        
    return json.dumps(reports, ensure_ascii=False)

# 执行聚合
patient_test_level = df_merged.groupby([
    '患者ID', 
    '【检验报告.检验大类名称】'
]).apply(build_patient_category_series, include_groups=False).reset_index()

# 强制重命名最后一列
patient_test_level.rename(columns={patient_test_level.columns[-1]: '时序结果集合'}, inplace=True)

# 6. 长表转宽表 (Pivot)
df_pivot = patient_test_level.pivot(
    index='患者ID', 
    columns='【检验报告.检验大类名称】', 
    values='时序结果集合'
).reset_index()

# 7. 合并基础表并导出
final_df = pd.merge(df_basic, df_pivot, on='患者ID', how='left')
final_df.to_csv('/data1/jianf/Excel work/processed_patient_data_with_units.csv', index=False, encoding='utf-8-sig')

print(f"带单位的数据清洗成功！共生成患者样本 {len(final_df)} 例。")