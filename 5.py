import pandas as pd
import json
import requests
import re

# ==========================================
# 步骤 1 & 2：数据读取与文本组装 (保持不变)
# ==========================================
df = pd.read_excel('/data1/jianf/Excel work/ecmo-0015301694.xlsx', header=1)
df['患者ID'] = df['患者ID'].ffill()

test_patient_id = df['患者ID'].dropna().unique()[0]
print(f"当前测试患者ID: {test_patient_id}")

df_patient = df[df['患者ID'] == test_patient_id]

if '【病历章节内容.章节名称】' in df.columns and '【病历章节内容.章节内容】' in df.columns:
    df_text = df_patient[['【病历章节内容.章节名称】', '【病历章节内容.章节内容】']].dropna()
else:
    raise KeyError("未找到【病历章节内容】相关的列，请检查表头。")

target_sections = [
    '主诉', '现病史', '既往史', '个人史', '家族史', 
    '初步诊断', '入院诊断', '出院诊断', '病例特点', '诊断依据', 
    '入院时主要症状及体征', '住院诊疗经过', '出院情况'
]

compiled_text = ""
for _, row in df_text.iterrows():
    section_name = str(row['【病历章节内容.章节名称】']).strip()
    content = str(row['【病历章节内容.章节内容】']).strip()
    
    if content and content.lower() != 'nan' and any(target in section_name for target in target_sections):
        compiled_text += f"【{section_name}】\n{content}\n\n"

# ==========================================
# 步骤 3：构建溯源型大语言模型提示词 (Evidence-based Prompt)
# ==========================================
# 采用嵌套 JSON 结构，强制模型在给出结论的同时摘录原文
prompt = f"""你是一个严谨的资深临床医学专家。请仔细阅读以下患者的电子病历文本，提取目标疾病的确诊状态及既往史。

【严格执行以下规则】
1. 结果只能从 ["是", "否", "未提及"] 中选择。
2. 对于每一个判断，必须在"证据"字段中原封不动地摘录支撑该判断的病历原文。如果判断为"未提及"或"否"，请在"证据"字段填写"无"。
3. 必须严格输出纯JSON格式，严禁包含任何Markdown标记（如```json）或分析过程。

【目标 JSON 结构示例】
{{
  "急性呼吸窘迫": {{"判断": "是", "证据": "患者出现严重的急性呼吸窘迫综合征..."}},
  "高血压史": {{"判断": "未提及", "证据": "无"}}
}}

【需要提取的目标字段】
- 急性呼吸窘迫
- 急性呼吸衰竭
- 心脏骤停
- 心源性休克
- 心肌炎
- 心肌梗死
- 其它诊断 (除上述疾病外的其他明确诊断)
- 高血压史
- 糖尿病史
- 心脏病史
- 脑血管病史

【病历文本】
{compiled_text}
"""

# ==========================================
# 步骤 4：调用本地 Ollama API (流式请求 + 完整捕获)
# ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b" 

payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": True,  # 保持流式以规避死锁
    "temperature": 0.0 # 设定温度值为0，确保医学信息提取的确定性与一致性
}

print(f"\n正在调用 {MODEL_NAME} 进行抽取与溯源计算...\n")

try:
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    
    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            word = chunk.get("response", "")
            full_response += word
            
    # ==========================================
    # 步骤 5：JSON 清洗与解析
    # ==========================================
    # 正则清洗可能存在的非法包裹符
    clean_json_str = re.sub(r"```json|```", "", full_response).strip()
    
    extraction_result = json.loads(clean_json_str)
    
    print("--- 结构化提取结果 ---")
    for condition, details in extraction_result.items():
        status = details.get('判断', '未知')
        evidence = details.get('证据', '无')
        print(f"[{condition}] -> 状态: {status} | 证据: {evidence}")

except requests.exceptions.RequestException as e:
    print(f"网络层异常: {e}")
except json.JSONDecodeError as e:
    print(f"JSON 解析失败，模型未能输出标准格式。原生文本为:\n{full_response}")
except Exception as e:
    print(f"系统异常: {e}")