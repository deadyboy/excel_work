import pandas as pd
import json
import requests
import re

# ==========================================
# 步骤 1：读取数据并锁定单一测试患者
# ==========================================
df = pd.read_excel('/data1/jianf/Excel work/ecmo-0015301694.xlsx', header=1)
df['患者ID'] = df['患者ID'].ffill()

# 选取第一个非空的患者ID作为测试用例
test_patient_id = df['患者ID'].dropna().unique()[0]
print(f"当前测试患者ID: {test_patient_id}")

df_patient = df[df['患者ID'] == test_patient_id]

# ==========================================
# 步骤 2：清洗并合并该患者的核心病历文本 (带召回率监控机制)
# ==========================================
if '【病历章节内容.章节名称】' in df.columns and '【病历章节内容.章节内容】' in df.columns:
    df_text = df_patient[['【病历章节内容.章节名称】', '【病历章节内容.章节内容】']].dropna()
else:
    raise KeyError("未找到【病历章节内容】相关的列，请检查表头。")

target_sections = [
    '主诉', '现病史', '既往史', '个人史', '家族史', 
    '初步诊断', '入院诊断', '出院诊断', '病例特点', '诊断依据', 
    '入院时主要症状及体征', '住院诊疗经过', '出院情况', '诊疗计划','鉴别诊断',
    '特殊检查及重要会诊','住院诊疗经过','出院情况'
]

compiled_text = ""
extracted_log = [] # 记录提取命中的章节及长度

for _, row in df_text.iterrows():
    section_name = str(row['【病历章节内容.章节名称】']).strip()
    content = str(row['【病历章节内容.章节内容】']).strip()
    
    # 过滤空内容，且判断是否在目标列表中
    if content and content.lower() != 'nan' and any(target in section_name for target in target_sections):
        compiled_text += f"【{section_name}】\n{content}\n\n"
        # 记录命中明细
        extracted_log.append(f"命中章节: {section_name.ljust(15)} | 提取字符数: {len(content)}")

# --- 调试监控输出 ---
print("\n" + "="*40)
print(" 文本召回质量检测报告 (Recall Report) ")
print("="*40)
if not extracted_log:
    print("警告：该患者未能匹配到任何目标章节！请检查底层数据是否包含所列列头。")
else:
    for log in extracted_log:
        print(log)
print("="*40)

# 强制将模型即将“读”到的纯文本上下文导出为本地 TXT 文件，用于人工交叉验证
debug_file_name = f"/data1/jianf/Excel work/debug_patient_{test_patient_id}_context.txt"
with open(debug_file_name, "w", encoding="utf-8") as f:
    f.write(compiled_text)
    
print(f"完整上下文已物理落盘至当前目录: {debug_file_name}")
print("在继续调用大模型前，请打开该 txt 文件。该文件内的文字，就是模型能获取到的绝对信息边界。\n")

# ==========================================
# 步骤 3：构建大语言模型提示词 (Prompt Engineering)
# ==========================================
# 采用 Zero-Shot JSON 强制输出格式，确保后续能够直接入库
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
{compiled_text}
"""

# ==========================================
# 步骤 4：调用本地 Ollama API
# ==========================================
# 假设您的 Ollama 运行在默认端口，调用的模型名称请根据您实际加载的模型进行替换（如 qwen2:72b-instruct）
# ==========================================
    # 步骤 4：调用本地 Ollama API (流式调试版)
    # ==========================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b" 
    
payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": True,  # 【关键修改】开启流式输出，实时监控推理状态
    "temperature": 0.0 # 设定温度值为0，确保医学信息提取的确定性与一致性
    # "format": "json" # 【关键修改】暂时注释掉 JSON 强制约束，测试是否为此导致的死循环
}

print(f"正在调用 Ollama 模型 ({MODEL_NAME})，请观察是否有文字实时输出...")
try:
        # 增加流式请求参数
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    
    print("\n--- 模型实时输出 ---")
    full_response = ""
    
    # 逐行读取流式响应
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            word = chunk.get("response", "")
            full_response += word
            # 实时打印到控制台，不换行，且立即刷新缓冲区
            print(word, end="", flush=True)
            
    print("\n\n--- 推理结束 ---")
        
except requests.exceptions.RequestException as e:
    print(f"\nOllama 请求出现网络级异常: {e}")