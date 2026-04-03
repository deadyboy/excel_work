import pandas as pd
import json
import requests
import re
import os
from tqdm import tqdm
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# ==========================================
# 全局配置参数
# ==========================================
INPUT_EXCEL_PATH = '/data1/jianf/Excel work/ecmo-0015301694.xlsx'
LLM_RAW_OUTPUT_PATH = '/data1/jianf/Excel work/llm_raw_results.jsonl'
DEBUG_DIR = '/data1/jianf/Excel work/debug_contexts'
FINAL_CSV_PATH = '/data1/jianf/Excel work/processed_patient_data_ultimate.csv'

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"   # 请替换为实际使用的模型名称

# 确保调试目录存在
os.makedirs(DEBUG_DIR, exist_ok=True)

# ==========================================
# 辅助函数定义
# ==========================================
def clean_id(series):
    """强制清洗主键，剔除浮点转换带来的 .0 以及多余空格"""
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def build_patient_category_series(group):
    """构建带单位的检验时序字典"""
    reports = []
    for report_id, sub_group in group.groupby('【检验结果.检验报告号】'):
        time_val = sub_group['【检验报告.报告时间】'].iloc[0]
        report_dict = {"时间": str(time_val)}
        
        for _, row in sub_group.iterrows():
            item_name = str(row['【检验结果.检验项目中文名称】']).strip()
            item_val = str(row['【检验结果.检验项目结果】']).strip()
            item_unit = str(row['【检验结果.检验项目结果单位】']).strip()
            
            if item_name and item_name.lower() != 'nan':
                if item_unit and item_unit.lower() not in ['nan', 'none', '']:
                    final_val = f"{item_val} {item_unit}"
                else:
                    final_val = item_val
                report_dict[item_name] = final_val
                
        reports.append(report_dict)
    return json.dumps(reports, ensure_ascii=False, cls=NpEncoder)

def build_all_tests_series(group):
    """构建全量检验时序字典（含检验大类名称）"""
    reports = []
    for report_id, sub_group in group.groupby('【检验结果.检验报告号】'):
        time_val = sub_group['【检验报告.报告时间】'].iloc[0]
        category_val = sub_group['【检验报告.检验大类名称】'].iloc[0]
        report_dict = {"时间": str(time_val), "检验大类": str(category_val)}
        
        for _, row in sub_group.iterrows():
            item_name = str(row['【检验结果.检验项目中文名称】']).strip()
            item_val = str(row['【检验结果.检验项目结果】']).strip()
            item_unit = str(row['【检验结果.检验项目结果单位】']).strip()
            
            if item_name and item_name.lower() != 'nan':
                if item_unit and item_unit.lower() not in ['nan', 'none', '']:
                    final_val = f"{item_val} {item_unit}"
                else:
                    final_val = item_val
                report_dict[item_name] = final_val
                
        reports.append(report_dict)
    return json.dumps(reports, ensure_ascii=False, cls=NpEncoder)

def extract_doc_records_by_keyword(df, pid_col, date_col, name_col, content_col,
                                   keywords, col_label):
    """
    从病历文书或病历章节内容中，按关键词过滤并按患者聚合为 JSON 列表。
    keywords: list[str]，满足其中任意一个即命中。
    返回 DataFrame，列为 [pid_col, col_label]。
    """
    if name_col not in df.columns or content_col not in df.columns:
        return pd.DataFrame(columns=[pid_col, col_label])

    mask = df[name_col].astype(str).str.contains('|'.join(keywords), na=False)
    df_filtered = df[mask][[pid_col, date_col, name_col, content_col]].dropna(subset=[pid_col]).copy()
    df_filtered = df_filtered[
        df_filtered[content_col].astype(str).str.strip().str.lower() != 'nan'
    ]

    if df_filtered.empty:
        return pd.DataFrame(columns=[pid_col, col_label])

    def _agg(grp):
        records = []
        for _, row in grp.iterrows():
            records.append({
                "时间": str(row[date_col]) if pd.notna(row[date_col]) else "",
                "文档名称": str(row[name_col]),
                "内容": str(row[content_col]).strip()
            })
        return json.dumps(records, ensure_ascii=False, cls=NpEncoder)

    result = df_filtered.groupby(pid_col).apply(_agg, include_groups=False).reset_index()
    result.columns = [pid_col, col_label]
    return result

# ==========================================
# 阶段一：结构化数据与文本切片预处理
# ==========================================
print(">>> 阶段一：结构化数据预处理中...")
df = pd.read_excel(INPUT_EXCEL_PATH, header=1)
df['患者ID'] = df['患者ID'].ffill()

# 1. 基础信息表
basic_cols = ['患者ID', '【患者基本信息.病人姓名】', '【患者基本信息.病人性别】', 
              '【住院登记.病人年龄】', '【住院登记.入院时间】', '【住院登记.出院时间】']
df_basic = df[basic_cols].dropna(subset=['患者ID']).drop_duplicates(subset=['患者ID'], keep='first')

# 2. 检验结果长宽表转换（按大类分列，原有逻辑保留）
df_report_meta = df[['【检验报告.检验报告号】', '【检验报告.检验大类名称】', '【检验报告.报告时间】',
                      '【检验报告.样本类型名称】']].dropna(subset=['【检验报告.检验报告号】']).copy()
df_report_meta['【检验报告.检验报告号】'] = clean_id(df_report_meta['【检验报告.检验报告号】'])
df_report_meta = df_report_meta.drop_duplicates(subset=['【检验报告.检验报告号】'], keep='first')

df_report_items = df[['患者ID', '【检验结果.检验报告号】', '【检验结果.检验项目中文名称】', 
                      '【检验结果.检验项目结果】', '【检验结果.检验项目结果单位】']].dropna(
                          subset=['【检验结果.检验报告号】', '【检验结果.检验项目中文名称】']).copy()
df_report_items['【检验结果.检验报告号】'] = clean_id(df_report_items['【检验结果.检验报告号】'])

df_merged = pd.merge(df_report_items, df_report_meta,
                     left_on='【检验结果.检验报告号】', right_on='【检验报告.检验报告号】', how='inner')

if not df_merged.empty:
    patient_test_level = df_merged.groupby(['患者ID', '【检验报告.检验大类名称】']).apply(
        build_patient_category_series, include_groups=False).reset_index()
    patient_test_level.rename(columns={patient_test_level.columns[-1]: '时序结果集合'}, inplace=True)
    df_pivot = patient_test_level.pivot(index='患者ID', columns='【检验报告.检验大类名称】',
                                        values='时序结果集合').reset_index()
else:
    df_pivot = pd.DataFrame(columns=['患者ID'])

# 3. 提取"主要化验结果"静态文本
if '【病历章节内容.章节名称】' in df.columns:
    mask_lab_text = df['【病历章节内容.章节名称】'].astype(str).str.strip() == '主要化验结果'
    df_lab_text = df[mask_lab_text][['患者ID', '【病历章节内容.章节内容】']].dropna()
    df_lab_text_grouped = df_lab_text.groupby('患者ID')['【病历章节内容.章节内容】'].apply(
        lambda x: '\n---\n'.join(x.astype(str).str.strip())
    ).reset_index()
    df_lab_text_grouped.rename(columns={'【病历章节内容.章节内容】': '主要化验结果文本'}, inplace=True)
else:
    df_lab_text_grouped = pd.DataFrame(columns=['患者ID'])

# ==========================================
# 阶段一（扩展）：新增列提取
# ==========================================

# 4. 所有APACHE评分
# 优先从病历章节内容中检索含"APACHE"的章节，再从病历文书中检索
print(">>> 提取所有APACHE评分...")
df_apache_section = pd.DataFrame(columns=['患者ID', '所有APACHE评分'])
df_apache_doc = pd.DataFrame(columns=['患者ID', '所有APACHE评分'])

if '【病历章节内容.章节名称】' in df.columns and '【病历章节内容.章节内容】' in df.columns:
    mask_apache_sec = df['【病历章节内容.章节名称】'].astype(str).str.contains('APACHE', case=False, na=False)
    df_tmp = df[mask_apache_sec][['患者ID', '【病历章节内容.文档提交时间】',
                                   '【病历章节内容.章节名称】', '【病历章节内容.章节内容】']].dropna(subset=['患者ID'])
    df_tmp = df_tmp[df_tmp['【病历章节内容.章节内容】'].astype(str).str.strip().str.lower() != 'nan']
    if not df_tmp.empty:
        def _agg_apache_sec(grp):
            recs = []
            for _, row in grp.iterrows():
                recs.append({
                    "时间": str(row['【病历章节内容.文档提交时间】']) if pd.notna(row['【病历章节内容.文档提交时间】']) else "",
                    "章节": str(row['【病历章节内容.章节名称】']),
                    "内容": str(row['【病历章节内容.章节内容】']).strip()
                })
            return json.dumps(recs, ensure_ascii=False, cls=NpEncoder)
        df_apache_section = df_tmp.groupby('患者ID').apply(
            _agg_apache_sec, include_groups=False).reset_index()
        df_apache_section.columns = ['患者ID', '所有APACHE评分']

if '【病历文书.文档名称】' in df.columns and '【病历文书.纯文本存放内容】' in df.columns:
    df_apache_doc = extract_doc_records_by_keyword(
        df, '患者ID', '【病历文书.文档日期】', '【病历文书.文档名称】',
        '【病历文书.纯文本存放内容】', ['APACHE'], '所有APACHE评分_文书')
    # 合并两个来源
    if not df_apache_section.empty and not df_apache_doc.empty:
        df_apache_section = df_apache_section.rename(columns={'所有APACHE评分': '_apache_sec'})
        df_apache_doc = df_apache_doc.rename(columns={'所有APACHE评分_文书': '_apache_doc'})
        df_apache_merged = pd.merge(df_apache_section, df_apache_doc, on='患者ID', how='outer')
        def _merge_apache(row):
            parts = []
            for c in ['_apache_sec', '_apache_doc']:
                v = row.get(c)
                if pd.notna(v) and str(v).lower() not in ['nan', 'none', '']:
                    parts.append(v)
            return ' | '.join(parts) if parts else None
        df_apache_merged['所有APACHE评分'] = df_apache_merged.apply(_merge_apache, axis=1)
        df_apache = df_apache_merged[['患者ID', '所有APACHE评分']]
    elif not df_apache_section.empty:
        df_apache = df_apache_section
    elif not df_apache_doc.empty:
        df_apache_doc = df_apache_doc.rename(columns={'所有APACHE评分_文书': '所有APACHE评分'})
        df_apache = df_apache_doc
    else:
        df_apache = pd.DataFrame(columns=['患者ID', '所有APACHE评分'])
else:
    df_apache = df_apache_section if not df_apache_section.empty else pd.DataFrame(columns=['患者ID', '所有APACHE评分'])

# 5. 所有检验（全量，不分大类）
print(">>> 提取所有检验...")
if not df_merged.empty:
    df_all_tests = df_merged.groupby('患者ID').apply(
        build_all_tests_series, include_groups=False).reset_index()
    df_all_tests.columns = ['患者ID', '所有检验']
else:
    df_all_tests = pd.DataFrame(columns=['患者ID', '所有检验'])

# 6. 所有血培养
# 策略：直接从检验报告元数据出发筛选血培养报告号，不依赖 df_merged（inner join 检验结果），
#       避免遗漏仅有微生物结果而无常规检验项目的血培养报告。
# 筛选条件：检验大类名称直接匹配"血培养"，
#           或者（样本类型含"血" AND 检验大类含"微生物"或"培养"）。
print(">>> 提取所有血培养...")

blood_culture_report_mask = (
    (df_report_meta['【检验报告.检验大类名称】'].astype(str).str.contains('血培养', na=False)) |
    (
        (df_report_meta['【检验报告.样本类型名称】'].astype(str).str.contains('血', na=False)) &
        (df_report_meta['【检验报告.检验大类名称】'].astype(str).str.contains('微生物|培养', na=False))
    )
)
df_bc_reports = df_report_meta[blood_culture_report_mask].copy()

# 关联患者ID（从原始表中取检验报告号→患者ID映射）
report_to_pid_bc = df[['患者ID', '【检验报告.检验报告号】']].dropna(
    subset=['【检验报告.检验报告号】']).copy()
report_to_pid_bc['【检验报告.检验报告号】'] = clean_id(report_to_pid_bc['【检验报告.检验报告号】'])
report_to_pid_bc = report_to_pid_bc.drop_duplicates(subset=['【检验报告.检验报告号】'])

df_bc_reports = pd.merge(df_bc_reports, report_to_pid_bc, on='【检验报告.检验报告号】', how='inner')

# 准备微生物结果和药敏表（供血培养和药敏共用）
microbio_result_cols = ['【检验微生物结果.检验报告号】', '【检验微生物结果.细菌项目名称】',
                         '【检验微生物结果.综合评价】']
microbio_susc_cols = ['【检验微生物药敏.检验报告号】', '【检验微生物药敏.细菌项目代码】',
                       '【检验微生物药敏.抗菌药物中文名称】', '【检验微生物药敏.最低抑菌浓度】',
                       '【检验微生物药敏.药敏结果】']

available_microbio_result_cols = [c for c in microbio_result_cols if c in df.columns]
available_microbio_susc_cols = [c for c in microbio_susc_cols if c in df.columns]

df_microbio_result = pd.DataFrame()
df_microbio_susc = pd.DataFrame()

if available_microbio_result_cols:
    df_microbio_result = df[available_microbio_result_cols].dropna(
        subset=['【检验微生物结果.检验报告号】']).copy()
    df_microbio_result['【检验微生物结果.检验报告号】'] = clean_id(
        df_microbio_result['【检验微生物结果.检验报告号】'])
    df_microbio_result = df_microbio_result.drop_duplicates()

if available_microbio_susc_cols:
    df_microbio_susc = df[available_microbio_susc_cols].dropna(
        subset=['【检验微生物药敏.检验报告号】']).copy()
    df_microbio_susc['【检验微生物药敏.检验报告号】'] = clean_id(
        df_microbio_susc['【检验微生物药敏.检验报告号】'])
    df_microbio_susc = df_microbio_susc.drop_duplicates()

# 准备一个按报告号索引的常规检验结果查找表（left join，允许为空）
# 注意：同一个报告号可能有多条检验结果项目，因此 index 允许重复，loc[[id]] 会正确返回所有行
df_report_items_indexed = df_report_items.set_index('【检验结果.检验报告号】') if not df_report_items.empty else pd.DataFrame()

def build_blood_culture_for_patient(patient_reports, df_report_items_indexed, df_microbio_result, df_microbio_susc):
    """构建血培养JSON：从检验报告出发，可选关联常规检验结果、细菌鉴定、药敏"""
    records = []
    if patient_reports.empty:
        return None
    for _, rpt_row in patient_reports.iterrows():
        report_id = rpt_row['【检验报告.检验报告号】']
        time_val = rpt_row['【检验报告.报告时间】']
        sample_type = rpt_row['【检验报告.样本类型名称】']
        category = rpt_row['【检验报告.检验大类名称】']
        entry = {"时间": str(time_val), "样本类型": str(sample_type), "检验大类": str(category)}

        # 常规检验项目（可能为空——纯微生物培养报告无常规项目）
        routine_results = {}
        if not df_report_items_indexed.empty and report_id in df_report_items_indexed.index:
            items = df_report_items_indexed.loc[[report_id]]
            for _, item_row in items.iterrows():
                item_name = str(item_row['【检验结果.检验项目中文名称】']).strip()
                item_val = str(item_row['【检验结果.检验项目结果】']).strip()
                item_unit = str(item_row['【检验结果.检验项目结果单位】']).strip()
                if item_name and item_name.lower() != 'nan':
                    if item_unit and item_unit.lower() not in ['nan', 'none', '']:
                        routine_results[item_name] = f"{item_val} {item_unit}"
                    else:
                        routine_results[item_name] = item_val
        if routine_results:
            entry['常规结果'] = routine_results

        # 微生物细菌鉴定结果
        if not df_microbio_result.empty and '【检验微生物结果.检验报告号】' in df_microbio_result.columns:
            rpt_bacteria = df_microbio_result[
                df_microbio_result['【检验微生物结果.检验报告号】'] == str(report_id)
            ]
            bacteria_list = []
            for _, br in rpt_bacteria.iterrows():
                b_name = str(br.get('【检验微生物结果.细菌项目名称】', '')).strip()
                b_eval = str(br.get('【检验微生物结果.综合评价】', '')).strip()
                if b_name and b_name.lower() != 'nan':
                    bacteria_list.append({"细菌": b_name, "综合评价": b_eval})
            if bacteria_list:
                entry['细菌鉴定'] = bacteria_list

        # 药敏结果
        if not df_microbio_susc.empty and '【检验微生物药敏.检验报告号】' in df_microbio_susc.columns:
            rpt_susc = df_microbio_susc[
                df_microbio_susc['【检验微生物药敏.检验报告号】'] == str(report_id)
            ]
            susc_list = []
            for _, sr in rpt_susc.iterrows():
                drug = str(sr.get('【检验微生物药敏.抗菌药物中文名称】', '')).strip()
                mic = str(sr.get('【检验微生物药敏.最低抑菌浓度】', '')).strip()
                result = str(sr.get('【检验微生物药敏.药敏结果】', '')).strip()
                if drug and drug.lower() != 'nan':
                    susc_list.append({"药物": drug, "MIC": mic, "结果": result})
            if susc_list:
                entry['药敏'] = susc_list

        records.append(entry)
    return json.dumps(records, ensure_ascii=False, cls=NpEncoder) if records else None

if not df_bc_reports.empty:
    blood_culture_records = []
    for pid, grp in df_bc_reports.groupby('患者ID'):
        val = build_blood_culture_for_patient(grp, df_report_items_indexed, df_microbio_result, df_microbio_susc)
        if val:
            blood_culture_records.append({"患者ID": pid, "所有血培养": val})
    df_blood_culture = pd.DataFrame(blood_culture_records) if blood_culture_records else pd.DataFrame(columns=['患者ID', '所有血培养'])
else:
    df_blood_culture = pd.DataFrame(columns=['患者ID', '所有血培养'])

# 7. 所有输血（从住院医嘱过滤血液制品相关项目）
print(">>> 提取所有输血记录...")
BLOOD_PRODUCT_KEYWORDS = ['输血', '红细胞', '血浆', '血小板', '全血', '冷沉淀', '白蛋白',
                           '血液', '悬浮红', '洗涤红', '新鲜冰冻']
order_cols = ['患者ID', '【住院医嘱.开单时间】', '【住院医嘱.医嘱项目名称】',
              '【住院医嘱.医嘱项目名称（通用名）】', '【住院医嘱.单次剂量】',
              '【住院医嘱.单次剂量单位】', '【住院医嘱.给药途径名称】', '【住院医嘱.数量】',
              '【住院医嘱.数量单位】', '【住院医嘱.医嘱状态名称】']
available_order_cols = [c for c in order_cols if c in df.columns]

df_orders = pd.DataFrame(columns=['患者ID', '所有输血'])
if '【住院医嘱.医嘱项目名称】' in df.columns:
    blood_order_mask = df['【住院医嘱.医嘱项目名称】'].astype(str).str.contains(
        '|'.join(BLOOD_PRODUCT_KEYWORDS), na=False)
    if '【住院医嘱.医嘱项目名称（通用名）】' in df.columns:
        blood_order_mask |= df['【住院医嘱.医嘱项目名称（通用名）】'].astype(str).str.contains(
            '|'.join(BLOOD_PRODUCT_KEYWORDS), na=False)
    df_blood_orders = df[blood_order_mask][available_order_cols].dropna(subset=['患者ID']).copy()

    if not df_blood_orders.empty:
        def _safe_str(val):
            """将可能为NaN的值安全转为字符串，NaN返回空字符串"""
            return '' if pd.isna(val) else str(val)

        def _agg_blood_orders(grp):
            recs = []
            for _, row in grp.iterrows():
                dose = _safe_str(row.get('【住院医嘱.单次剂量】'))
                dose_unit = _safe_str(row.get('【住院医嘱.单次剂量单位】'))
                qty = _safe_str(row.get('【住院医嘱.数量】'))
                qty_unit = _safe_str(row.get('【住院医嘱.数量单位】'))
                rec = {
                    "时间": _safe_str(row.get('【住院医嘱.开单时间】')),
                    "医嘱名称": _safe_str(row.get('【住院医嘱.医嘱项目名称】')),
                    "通用名": _safe_str(row.get('【住院医嘱.医嘱项目名称（通用名）】')),
                    "剂量": f"{dose}{dose_unit}" if dose else "",
                    "途径": _safe_str(row.get('【住院医嘱.给药途径名称】')),
                    "数量": f"{qty}{qty_unit}" if qty else "",
                    "状态": _safe_str(row.get('【住院医嘱.医嘱状态名称】'))
                }
                recs.append(rec)
            return json.dumps(recs, ensure_ascii=False, cls=NpEncoder)
        df_orders = df_blood_orders.groupby('患者ID').apply(
            _agg_blood_orders, include_groups=False).reset_index()
        df_orders.columns = ['患者ID', '所有输血']

# 8. 所有药敏（来自检验微生物药敏，关联微生物结果和检验报告）
print(">>> 提取所有药敏记录...")
df_all_susc = pd.DataFrame(columns=['患者ID', '所有药敏'])
if not df_microbio_susc.empty and '【检验微生物药敏.检验报告号】' in df_microbio_susc.columns:
    # 通过检验报告号关联患者ID
    report_to_pid = df[['患者ID', '【检验报告.检验报告号】']].dropna(
        subset=['【检验报告.检验报告号】']).copy()
    report_to_pid['【检验报告.检验报告号】'] = clean_id(report_to_pid['【检验报告.检验报告号】'])
    report_to_pid = report_to_pid.drop_duplicates(subset=['【检验报告.检验报告号】'])

    report_time = df_report_meta[['【检验报告.检验报告号】', '【检验报告.报告时间】']].copy()

    df_susc_with_pid = pd.merge(df_microbio_susc, report_to_pid,
                                 left_on='【检验微生物药敏.检验报告号】',
                                 right_on='【检验报告.检验报告号】', how='inner')
    df_susc_with_pid = pd.merge(df_susc_with_pid, report_time,
                                 left_on='【检验微生物药敏.检验报告号】',
                                 right_on='【检验报告.检验报告号】', how='left',
                                 suffixes=('', '_time'))

    if not df_susc_with_pid.empty:
        def _agg_susc(grp):
            # 按报告号分组
            recs = []
            for report_id, sub in grp.groupby('【检验微生物药敏.检验报告号】'):
                time_val = sub['【检验报告.报告时间】'].iloc[0] if '【检验报告.报告时间】' in sub.columns else ''
                # 按细菌项目代码再分组
                drug_list = []
                for _, row in sub.iterrows():
                    drug = str(row.get('【检验微生物药敏.抗菌药物中文名称】', '')).strip()
                    mic = str(row.get('【检验微生物药敏.最低抑菌浓度】', '')).strip()
                    result = str(row.get('【检验微生物药敏.药敏结果】', '')).strip()
                    bacteria_code = str(row.get('【检验微生物药敏.细菌项目代码】', '')).strip()
                    if drug and drug.lower() != 'nan':
                        drug_list.append({"细菌代码": bacteria_code, "药物": drug, "MIC": mic, "结果": result})
                recs.append({"时间": str(time_val), "报告号": str(report_id), "药敏列表": drug_list})
            return json.dumps(recs, ensure_ascii=False, cls=NpEncoder)

        df_all_susc = df_susc_with_pid.groupby('患者ID').apply(
            _agg_susc, include_groups=False).reset_index()
        df_all_susc.columns = ['患者ID', '所有药敏']

# 9 & 10. ECMO上机记录 / ECMO下机记录
# 来源1：病历文书
print(">>> 提取ECMO上/下机记录...")
df_ecmo_on = pd.DataFrame(columns=['患者ID', 'ECMO上机记录'])
df_ecmo_off = pd.DataFrame(columns=['患者ID', 'ECMO下机记录'])

if '【病历文书.文档名称】' in df.columns and '【病历文书.纯文本存放内容】' in df.columns:
    df_ecmo_on_doc = extract_doc_records_by_keyword(
        df, '患者ID', '【病历文书.文档日期】', '【病历文书.文档名称】',
        '【病历文书.纯文本存放内容】',
        ['ECMO上机', 'ECMO建立', 'ECMO辅助', 'V-A ECMO开始', 'V-V ECMO开始', '体外膜肺上机'],
        'ECMO上机记录')
    df_ecmo_off_doc = extract_doc_records_by_keyword(
        df, '患者ID', '【病历文书.文档日期】', '【病历文书.文档名称】',
        '【病历文书.纯文本存放内容】',
        ['ECMO下机', 'ECMO撤机', 'ECMO停机', '体外膜肺下机'],
        'ECMO下机记录')
    df_ecmo_on = df_ecmo_on_doc if not df_ecmo_on_doc.empty else df_ecmo_on
    df_ecmo_off = df_ecmo_off_doc if not df_ecmo_off_doc.empty else df_ecmo_off

# 来源2：病历章节内容（补充上机/下机章节）
if '【病历章节内容.章节名称】' in df.columns and '【病历章节内容.章节内容】' in df.columns:
    for label, on_keywords, off_keywords in [
        ('ECMO上机记录', ['ECMO上机', 'ECMO建立', '体外膜肺上机'], None),
        ('ECMO下机记录', None, ['ECMO下机', 'ECMO撤机', '体外膜肺下机']),
    ]:
        kws = on_keywords if on_keywords else off_keywords
        mask_sec = df['【病历章节内容.章节名称】'].astype(str).str.contains('|'.join(kws), na=False)
        df_tmp_sec = df[mask_sec][['患者ID', '【病历章节内容.文档提交时间】',
                                    '【病历章节内容.章节名称】', '【病历章节内容.章节内容】']].dropna(subset=['患者ID'])
        df_tmp_sec = df_tmp_sec[df_tmp_sec['【病历章节内容.章节内容】'].astype(str).str.strip().str.lower() != 'nan']
        if not df_tmp_sec.empty:
            def _agg_sec(grp, lbl=label):
                recs = []
                for _, row in grp.iterrows():
                    recs.append({
                        "时间": str(row['【病历章节内容.文档提交时间】']) if pd.notna(row['【病历章节内容.文档提交时间】']) else "",
                        "章节": str(row['【病历章节内容.章节名称】']),
                        "内容": str(row['【病历章节内容.章节内容】']).strip()
                    })
                return json.dumps(recs, ensure_ascii=False, cls=NpEncoder)
            df_sec_result = df_tmp_sec.groupby('患者ID').apply(
                _agg_sec, include_groups=False).reset_index()
            df_sec_result.columns = ['患者ID', label]
            # 若之前已有来源1的数据，做外连接补充
            if label == 'ECMO上机记录':
                if df_ecmo_on.empty:
                    df_ecmo_on = df_sec_result
                else:
                    df_ecmo_on = pd.merge(df_ecmo_on, df_sec_result, on='患者ID', how='outer',
                                          suffixes=('', '_sec'))
                    if 'ECMO上机记录_sec' in df_ecmo_on.columns:
                        df_ecmo_on['ECMO上机记录'] = df_ecmo_on['ECMO上机记录'].combine_first(
                            df_ecmo_on['ECMO上机记录_sec'])
                        df_ecmo_on.drop(columns=['ECMO上机记录_sec'], inplace=True)
            else:
                if df_ecmo_off.empty:
                    df_ecmo_off = df_sec_result
                else:
                    df_ecmo_off = pd.merge(df_ecmo_off, df_sec_result, on='患者ID', how='outer',
                                           suffixes=('', '_sec'))
                    if 'ECMO下机记录_sec' in df_ecmo_off.columns:
                        df_ecmo_off['ECMO下机记录'] = df_ecmo_off['ECMO下机记录'].combine_first(
                            df_ecmo_off['ECMO下机记录_sec'])
                        df_ecmo_off.drop(columns=['ECMO下机记录_sec'], inplace=True)

# 11. 日常病程录
print(">>> 提取日常病程录...")
df_daily_progress = pd.DataFrame(columns=['患者ID', '日常病程录'])
if '【病历文书.文档名称】' in df.columns and '【病历文书.纯文本存放内容】' in df.columns:
    df_daily_progress = extract_doc_records_by_keyword(
        df, '患者ID', '【病历文书.文档日期】', '【病历文书.文档名称】',
        '【病历文书.纯文本存放内容】',
        ['日常病程录', '病程记录', '病程录'],
        '日常病程录')

# 12. 出院记录
print(">>> 提取出院记录...")
df_discharge = pd.DataFrame(columns=['患者ID', '出院记录'])
if '【病历文书.文档名称】' in df.columns and '【病历文书.纯文本存放内容】' in df.columns:
    df_discharge = extract_doc_records_by_keyword(
        df, '患者ID', '【病历文书.文档日期】', '【病历文书.文档名称】',
        '【病历文书.纯文本存放内容】',
        ['出院记录', '出院小结'],
        '出院记录')

# 13. 死亡记录
print(">>> 提取死亡记录...")
df_death = pd.DataFrame(columns=['患者ID', '死亡记录'])
if '【病历文书.文档名称】' in df.columns and '【病历文书.纯文本存放内容】' in df.columns:
    df_death = extract_doc_records_by_keyword(
        df, '患者ID', '【病历文书.文档日期】', '【病历文书.文档名称】',
        '【病历文书.纯文本存放内容】',
        ['死亡记录', '死亡病例讨论'],
        '死亡记录')

# ==========================================
# 阶段二：大语言模型推理与召回监控引擎
# ==========================================
print("\n>>> 阶段二：启动 LLM 遍历推理...")
target_sections = ['主诉', '现病史', '既往史', '个人史', '家族史', '初步诊断', '入院诊断', '出院诊断', 
                   '病例特点', '诊断依据', '入院时主要症状及体征', '住院诊疗经过', '出院情况']

all_patient_ids = df_basic['患者ID'].unique()

# 校验已处理的进度（断点续传机制）
processed_ids = set()
if os.path.exists(LLM_RAW_OUTPUT_PATH):
    with open(LLM_RAW_OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            processed_ids.add(json.loads(line).get("患者ID"))
    print(f"检测到历史暂存文件，已跳过 {len(processed_ids)} 名已处理患者。")

with open(LLM_RAW_OUTPUT_PATH, 'a', encoding='utf-8') as f_out:
    for pid in tqdm(all_patient_ids, desc="患者处理进度"):
        if pid in processed_ids:
            continue
            
        # 1. 召回层：组装该患者文本
        df_patient_text = df[(df['患者ID'] == pid) & (df['【病历章节内容.章节名称】'].notna()) & (df['【病历章节内容.章节内容】'].notna())]
        compiled_text = ""
        recall_log = []
        
        for _, row in df_patient_text.iterrows():
            section_name = str(row['【病历章节内容.章节名称】']).strip()
            content = str(row['【病历章节内容.章节内容】']).strip()
            if content and content.lower() != 'nan' and any(target in section_name for target in target_sections):
                compiled_text += f"【{section_name}】\n{content}\n\n"
                recall_log.append(f"命中: {section_name} ({len(content)}字)")
        
        # 将原始上下文落盘供审计
        with open(os.path.join(DEBUG_DIR, f"{pid}_context.txt"), "w", encoding="utf-8") as f_debug:
            f_debug.write(f"--- 召回日志 ---\n" + "\n".join(recall_log) + "\n\n--- 临床文本 ---\n" + compiled_text)
            
        # 若无可推理文本，记录全空结果
        if not compiled_text:
            empty_res = {"患者ID": pid, "error": "无有效临床文本，跳过推理"}
            f_out.write(json.dumps(empty_res, ensure_ascii=False, cls=NpEncoder) + '\n')
            f_out.flush()
            continue

        # 2. 推理层：调用 LLM
        prompt = f"""你是一个严谨的资深临床医学专家。请仔细阅读以下患者的电子病历文本，判断患者是否患有以下疾病或具有以下病史。

判断标准：
1. 必须基于文本中明确的诊断或描述进行判断。
2. 结果只能从 ["是", "否", "未提及"] 中选择。

目标提取字段：
- 是否急性呼吸窘迫
- 是否急性呼吸衰竭
- 是否心脏骤停
- 是否心源性休克
- 是否心肌炎
- 是否心肌梗死
- 是否有其它诊断 (除上述疾病外的其他明确诊断，如果有，请填写"是"，否则填写"否")
- 是否有高血压史
- 是否有糖尿病史
- 是否有心脏病史
- 是否有脑血管病史

请严格以JSON格式返回结果，JSON的键必须与上述目标字段完全一致。不要输出任何除了JSON以外的解释性文字！

病历文本：
{compiled_text}"""

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,  # 【关键修改】关闭流式输出，确保一次性获取完整响应
            "temperature": 0.0 # 设定温度值为0，确保医学信息提取的确定性与一致性
            # "format": "json" # 【关键修改】暂时注释掉 JSON 强制约束，测试是否为此导致的死循环
        }
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            result_text = response.json().get("response", "")
            
            # 【核心修正】：使用非贪婪/贪婪正则，提取首尾的 {} 内容，自动过滤掉模型可能生成的废话前缀和 Markdown 语法
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not match:
                raise ValueError(f"未能从模型输出中提取到有效的 JSON 结构。原始输出为: {result_text}")
            
            clean_json_str = match.group(0)
            extraction_result = json.loads(clean_json_str)
            
            # 强制注入患者主键
            extraction_result["患者ID"] = pid
            
            # 原生结果落盘（保留"未提及"）
            f_out.write(json.dumps(extraction_result, ensure_ascii=False, cls=NpEncoder) + '\n')
            f_out.flush()
            
        except json.JSONDecodeError as e:
            error_res = {"患者ID": pid, "error": f"JSON解析失败: {str(e)}。清洗后的字符串为: {clean_json_str}"}
            f_out.write(json.dumps(error_res, ensure_ascii=False, cls=NpEncoder) + '\n')
            f_out.flush()
        except Exception as e:
            error_res = {"患者ID": pid, "error": f"LLM推理或正则提取异常: {str(e)}"}
            f_out.write(json.dumps(error_res, ensure_ascii=False, cls=NpEncoder) + '\n')
            f_out.flush()

# ==========================================
# 阶段三：后处理映射与宽表归一化引擎
# ==========================================
print("\n>>> 阶段三：合并特征与最终宽表生成...")

# 读取暂存的 LLM 原生结果
llm_results_list = []
with open(LLM_RAW_OUTPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        llm_results_list.append(json.loads(line))

df_llm = pd.DataFrame(llm_results_list)

# 排除含有 error 的异常行，以免影响正常列格式
if 'error' in df_llm.columns:
    df_llm = df_llm[df_llm['error'].isna()].drop(columns=['error'])

# 核心后处理：将宽表中的 "未提及" 确定性映射为 "否"
llm_feature_cols = [col for col in df_llm.columns if col != '患者ID']
for col in llm_feature_cols:
    df_llm[col] = df_llm[col].replace("未提及", "否")

# 依次执行左连接 (Left Join)——原有列
final_df = pd.merge(df_basic, df_pivot, on='患者ID', how='left')
final_df = pd.merge(final_df, df_lab_text_grouped, on='患者ID', how='left')
final_df = pd.merge(final_df, df_llm, on='患者ID', how='left')

# 新增列的左连接
new_dfs = [
    df_apache,        # 所有APACHE评分
    df_all_tests,     # 所有检验
    df_blood_culture, # 所有血培养
    df_orders,        # 所有输血
    df_all_susc,      # 所有药敏
    df_ecmo_on,       # ECMO上机记录
    df_ecmo_off,      # ECMO下机记录
    df_daily_progress,# 日常病程录
    df_discharge,     # 出院记录
    df_death,         # 死亡记录
]
for extra_df in new_dfs:
    if not extra_df.empty and '患者ID' in extra_df.columns:
        final_df = pd.merge(final_df, extra_df, on='患者ID', how='left')

# 输出最终成果
final_df.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"数据清洗与多模态整合完毕。共生成样本 {len(final_df)} 例，特征维度扩充至 {len(final_df.columns)} 列。")
print(f"输出文件路径：{FINAL_CSV_PATH}")
