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

# 2. 检验结果长宽表转换
df_report_meta = df[['【检验报告.检验报告号】', '【检验报告.检验大类名称】', '【检验报告.报告时间】']].dropna(subset=['【检验报告.检验报告号】']).copy()
df_report_meta['【检验报告.检验报告号】'] = clean_id(df_report_meta['【检验报告.检验报告号】'])
df_report_meta = df_report_meta.drop_duplicates(subset=['【检验报告.检验报告号】'], keep='first')

df_report_items = df[['患者ID', '【检验结果.检验报告号】', '【检验结果.检验项目中文名称】', 
                      '【检验结果.检验项目结果】', '【检验结果.检验项目结果单位】']].dropna(subset=['【检验结果.检验报告号】', '【检验结果.检验项目中文名称】']).copy()
df_report_items['【检验结果.检验报告号】'] = clean_id(df_report_items['【检验结果.检验报告号】'])

df_merged = pd.merge(df_report_items, df_report_meta, left_on='【检验结果.检验报告号】', right_on='【检验报告.检验报告号】', how='inner')

if not df_merged.empty:
    patient_test_level = df_merged.groupby(['患者ID', '【检验报告.检验大类名称】']).apply(build_patient_category_series, include_groups=False).reset_index()
    patient_test_level.rename(columns={patient_test_level.columns[-1]: '时序结果集合'}, inplace=True)
    df_pivot = patient_test_level.pivot(index='患者ID', columns='【检验报告.检验大类名称】', values='时序结果集合').reset_index()
else:
    df_pivot = pd.DataFrame(columns=['患者ID'])

# 3. 提取“主要化验结果”静态文本
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
            
            # 原生结果落盘（保留“未提及”）
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

# 依次执行左连接 (Left Join)
final_df = pd.merge(df_basic, df_pivot, on='患者ID', how='left')
final_df = pd.merge(final_df, df_lab_text_grouped, on='患者ID', how='left')
final_df = pd.merge(final_df, df_llm, on='患者ID', how='left')

# 输出最终成果
final_df.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"数据清洗与多模态整合完毕。共生成样本 {len(final_df)} 例，特征维度扩充至 {len(final_df.columns)} 列。")
print(f"输出文件路径：{FINAL_CSV_PATH}")