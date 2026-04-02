import pandas as pd
import json
import numpy as np

# 1. 读取数据
df = pd.read_excel('/data1/jianf/Excel work/ecmo-0015301694.xlsx', header=1)

# 2. 规范化“检验报告号”（避免 52943198 vs 52943198.0 的不匹配）
def normalize_report_id(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if float(val).is_integer():
            return str(int(val))
        return str(val).strip()
    s = str(val).strip()
    if s.endswith('.0') and s.replace('.0', '').isdigit():
        return s[:-2]
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

# 3. 提取并去重患者基础信息
basic_cols = [
    '患者ID', 
    '【患者基本信息.病人姓名】', 
    '【患者基本信息.病人性别】', 
    '【住院登记.病人年龄】',
    '【住院登记.入院时间】',
    '【住院登记.出院时间】'
]
df_basic = df[basic_cols].dropna(subset=['患者ID']).drop_duplicates(subset=['患者ID'], keep='first')

# 4. 彻底分离并重建关系映射（规避物理行平铺错位）
df_report_meta = df[['患者ID', '【检验报告.检验报告号】', '【检验报告.检验大类名称】', '【检验报告.报告时间】']].dropna(
    subset=['【检验报告.检验报告号】']
)
df_report_meta['report_id'] = df_report_meta['【检验报告.检验报告号】'].apply(normalize_report_id)
df_report_meta = df_report_meta.dropna(subset=['report_id'])
df_report_meta = df_report_meta.drop_duplicates(subset=['患者ID', 'report_id'], keep='first')

df_report_items = df[['患者ID', '【检验结果.检验报告号】', '【检验结果.检验项目中文名称】', '【检验结果.检验项目结果】']].dropna(
    subset=['【检验结果.检验报告号】', '【检验结果.检验项目中文名称】']
)
df_report_items['report_id'] = df_report_items['【检验结果.检验报告号】'].apply(normalize_report_id)
df_report_items = df_report_items.dropna(subset=['report_id'])
# 同一报告内项目名重复时保留首条（该文件里重复项结果一致）
df_report_items = df_report_items.drop_duplicates(
    subset=['患者ID', 'report_id', '【检验结果.检验项目中文名称】'],
    keep='first'
)

df_merged = pd.merge(
    df_report_items,
    df_report_meta[['患者ID', 'report_id', '【检验报告.检验大类名称】', '【检验报告.报告时间】']],
    on=['患者ID', 'report_id'],
    how='inner'
)

# 4 & 5. 单层 GroupBy 结合原生字典序列化
def build_patient_category_series(group):
    report_entries = []
    # 在同一患者的同一大类下，按具体的“报告号”二次分组
    for report_id, sub_group in group.groupby('report_id'):
        time_val = sub_group['【检验报告.报告时间】'].iloc[0]
        report_dict = {"时间": str(time_val) if pd.notna(time_val) else ""}

        for _, row in sub_group.iterrows():
            item_name = str(row['【检验结果.检验项目中文名称】']).strip()
            item_val = str(row['【检验结果.检验项目结果】']).strip()
            if item_name and item_name != 'nan':
                report_dict[item_name] = item_val

        report_entries.append((time_val, report_dict))

    # 按时间排序（若时间为空则保持在后）
    def _sort_key(x):
        t = pd.to_datetime(x[0], errors='coerce')
        return (pd.isna(t), t)

    report_entries.sort(key=_sort_key)
    reports = [d for _, d in report_entries]
    return json.dumps(reports, ensure_ascii=False)

# 核心修正：仅执行基础的 reset_index()，不传入任何会导致类型冲突的参数
patient_test_level = df_merged.groupby([
    '患者ID', 
    '【检验报告.检验大类名称】'
]).apply(build_patient_category_series, include_groups=False).reset_index()

# 核心修正：无论 apply 隐式生成了什么列名（如 0 或 None），强制将最后一列重命名为目标字段
patient_test_level.rename(columns={patient_test_level.columns[-1]: '时序结果集合'}, inplace=True)

# 6. 长表转宽表 (Pivot)
df_pivot = patient_test_level.pivot(
    index='患者ID', 
    columns='【检验报告.检验大类名称】', 
    values='时序结果集合'
).reset_index()

# 7. 合并基础表并导出
final_df = pd.merge(df_basic, df_pivot, on='患者ID', how='left')
final_df.to_csv('/data1/jianf/Excel work/processed_patient_data_v2.csv', index=False, encoding='utf-8-sig')

print("特征工程构建完成。长宽表转换成功执行。")
