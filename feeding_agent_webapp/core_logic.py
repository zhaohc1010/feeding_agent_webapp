# -*- coding: utf-8 -*-
"""
核心逻辑模块
包含了所有的数据处理、模型预测和日志记录功能。
"""

# --- 基础库导入 ---
import os
import re
import warnings
from datetime import datetime

# --- 第三方库导入 ---
import pandas as pd
import docx
import joblib
import numpy as np
from openai import OpenAI
from sklearn.exceptions import InconsistentVersionWarning

# --- 从配置文件导入变量 ---
from config import LOG_FILE_PATH, MODEL_NAME, BASE_URL, LGBM_MODEL_PATH, KNOWLEDGE_BASE_PATH

# --- 管理警告信息 ---
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# ==============================================================================
#                               辅助功能
# ==============================================================================

def read_knowledge_base():
    """
    功能：读取指定的 .docx Word文档，并提取所有文本内容。
    """
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        # 仅在非开发环境打印错误，避免干扰日志
        return None
    try:
        document = docx.Document(KNOWLEDGE_BASE_PATH)
        return '\n'.join([para.text for para in document.paragraphs])
    except Exception as e:
        print(f"读取知识库文件时发生错误: {e}")
        return None


def read_historical_data():
    """
    功能：读取Excel日志文件，并返回最近10条记录作为字符串。
    """
    print("\n正在读取历史投喂记录...")
    if not os.path.exists(LOG_FILE_PATH):
        return "尚无历史投喂记录。"
    try:
        df = pd.read_excel(LOG_FILE_PATH)
        if df.empty:
            return "历史投喂记录文件为空。"
        return df.tail(10).to_string()
    except Exception as e:
        return f"读取历史记录时发生错误: {e}"


# ==============================================================================
#                               模型预测功能
# ==============================================================================

def predict_with_lightgbm(user_data):
    """
    使用加载的LightGBM模型进行预测。
    """
    print("\n正在使用 LightGBM 模型进行预测...")
    try:
        loaded_artifacts = joblib.load(LGBM_MODEL_PATH)
        model = loaded_artifacts['model']
        input_scaler = loaded_artifacts['input_scaler']
        output_scaler = loaded_artifacts['output_scaler']
    except FileNotFoundError:
        print(f"错误：LightGBM模型文件未找到！请检查路径：{LGBM_MODEL_PATH}")
        return None
    except Exception as e:
        print(f"错误：模型加载失败: {e}")
        return None

    feature_keys = [
        "month_day", "system_id", "average_water_temp", "average_do", "average_ph",
        "ammonia_nitrogen", "nitrite_nitrogen", "water_change_amount", "water_change_rate",
        "age_in_days", "weight", "shrimp_loading_cap", "water_body_capacity",
        "survival_rate", "avg_daily_weight_gain", "number_of_meals"
    ]

    # 检查是否缺少必要的键
    missing_keys = [key for key in feature_keys if key not in user_data]
    if missing_keys:
        print(f"错误：用户输入数据缺少LightGBM模型所需的字段: {missing_keys}")
        return None

    # 构建 DataFrame
    input_df = pd.DataFrame([user_data])[feature_keys]

    try:
        # 直接传入 input_df，让 input_scaler 自动处理类型转换（包括 system_id 字符串）
        input_data_scaled = input_scaler.transform(input_df)

        prediction_scaled = model.predict(input_data_scaled)
        prediction_scaled_reshaped = prediction_scaled.reshape(-1, 1)
        final_prediction = output_scaler.inverse_transform(prediction_scaled_reshaped)

        result = final_prediction[0][0]
        print(f"LightGBM 模型预测结果: {result:.4f} kg")
        return result

    except Exception as e:
        print(f"预测计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_from_formulas(user_data):
    """
    根据用户提供的硬编码公式计算投喂量。
    返回一个元组 (预测值, 计算依据说明)。
    """
    print("\n正在根据内置公式进行计算...")
    weight = user_data.get('weight')
    shrimp_loading_cap = user_data.get('shrimp_loading_cap')
    water_body_capacity = user_data.get('water_body_capacity')

    if not all([isinstance(weight, (int, float)),
                isinstance(shrimp_loading_cap, (int, float)),
                isinstance(water_body_capacity, (int, float))]):
        return None, "缺少计算所需的关键参数（体重、存塘量、水体容量）。"

    try:
        if weight == 0: raise ZeroDivisionError
        estimated_shrimp_count = shrimp_loading_cap * 500 * water_body_capacity / weight
    except ZeroDivisionError:
        return None, "错误：对虾体重为0，无法计算预估数量。"

    coeffs = None
    if 0.50 <= weight <= 0.60:
        coeffs = (0.144, 0.162)
    elif 0.98 <= weight <= 1.12:
        coeffs = (0.129, 0.134)
    elif 1.79 <= weight <= 2.12:
        coeffs = (0.097, 0.099)
    elif 2.72 <= weight <= 3.29:
        coeffs = (0.084, 0.085)
    elif 3.94 <= weight <= 5.10:
        coeffs = (0.065, 0.076)
    elif 5.32 <= weight <= 6.76:
        coeffs = (0.061, 0.072)
    elif 7.46 <= weight <= 8.93:
        coeffs = (0.058, 0.062)
    elif 9.62 <= weight <= 11.63:
        coeffs = (0.045, 0.052)
    elif 11.63 <= weight <= 13.51:
        coeffs = (0.040, 0.044)
    elif 13.51 <= weight <= 15.63:
        coeffs = (0.032, 0.040)
    elif 16.13 <= weight <= 16.67:
        coeffs = (0.032, 0.037)
    elif 17.80 <= weight <= 18.00:
        coeffs = (0.030, 0.030)
    elif 19.20 <= weight <= 19.40:
        coeffs = (0.025, 0.025)
    elif 20.00 <= weight <= 20.83:
        coeffs = (0.025, 0.025)

    if coeffs is None:
        return None, f"当前对虾体重 {weight}g 未匹配到任何预设的投喂系数范围。"

    low_bound = weight * coeffs[0] * estimated_shrimp_count * 0.001
    high_bound = weight * coeffs[1] * estimated_shrimp_count * 0.001
    average_prediction = (low_bound + high_bound) / 2

    explanation = (
        f"根据内置公式计算：\n"
        f"- 预估对虾数量: {estimated_shrimp_count:.0f} 尾\n"
        f"- 匹配体重范围: {weight}g, 使用系数 {coeffs[0]} 至 {coeffs[1]}\n"
        f"- 计算投喂量范围: {low_bound:.4f} kg 至 {high_bound:.4f} kg\n"
        f"- 取平均值作为预测结果。"
    )

    print(f"内置公式计算结果: {average_prediction:.4f} kg")
    return average_prediction, explanation


def get_final_decision_with_remarks(lgbm_pred, formula_pred_tuple, user_data, knowledge_content, historical_data_str,
                                    language='en'):
    """
    当用户提供了备注时，调用LLM来决定最终使用哪个预测结果。
    language: 'zh' 或 'en' (默认 en)
    """
    print(f"\n检测到用户备注，正在启动包含历史数据的智能决策模块 (语言: {language})...")
    formula_pred, formula_explanation = formula_pred_tuple
    try:
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=BASE_URL)

        # 核心修改：保留了您要求的 5 步逻辑，但添加了【输出要精简】的强指令
        if language == 'en':
            system_prompt = """
            You are a Chief Expert in shrimp farming. Your task is to be the final decision maker.

            Your decision process MUST follow these steps (think deeply, but keep the output CONCISE):
            1. Prioritize analyzing key information in the [User Remarks].
            2. Combine [Knowledge Base] and [Historical Records] to understand the impact of the remarks.
            3. Evaluate the two prediction values (Model vs Formula) to see which is more reasonable.
            4. Make a final choice. **Explain your reasoning CONCISELY** (summarize key points in 2-3 sentences, avoid lengthy analysis).
            5. At the end, you MUST strictly follow the format '【Final Feeding Amount】: XX.XX' on a new line.
            """
            user_intro = "Please make the final judgment based on the following info (Answer in English, keep it concise):"
            lgbm_label = "LightGBM Prediction"
            formula_label = "Formula Result"
            kb_label = "Knowledge Base"
            history_label = "History Logs"
            current_label = "Current Data"
            remarks_label = "User Remarks"
        else:
            system_prompt = """
            你是一位经验丰富的对虾养殖首席专家，任务是作为最终决策者。

            你的决策流程必须严格遵循以下步骤（请在内心进行深度思考，但**输出结果需精简**）：
            1. 优先分析【用户备注】中的关键信息。
            2. 结合【知识库全文】和【历史投喂记录】，理解备注信息对投喂量的影响。
            3. 综合所有信息，评估两个预测值哪个更合理，或提出一个更优的调整值。
            4. 做出最终选择。**请精简地总结你的理由**（不要输出冗长的分析过程，用2-3句话直接指出核心依据）。
            5. 在回答的最后，必须另起一行，并严格按照 '【最终投喂量】: XX.XX' 的格式给出你最终采纳的投喂量数值。
            """
            user_intro = "请根据以上所有信息，做出最终的投喂量裁决（请保持回答简练）："
            lgbm_label = "LightGBM模型预测值"
            formula_label = "内置公式计算结果"
            kb_label = "知识库全文"
            history_label = "历史投喂记录"
            current_label = "用户实时养殖数据"
            remarks_label = "用户备注"

        user_prompt = f"""
        【{lgbm_label}】: {lgbm_pred:.4f} kg

        【{formula_label}】: {formula_pred:.4f} kg
        (计算依据: {formula_explanation})
        ---

        【{kb_label}】:
        ---
        {knowledge_content[:800]}...
        ---

        【{history_label}】 (最近10条):
        ---
        {historical_data_str}
        ---

        【{current_label}】:
        ---
        {user_data}
        ---

        【{remarks_label}】:
        ---
        {user_data.get('remarks')}
        ---

        {user_intro}
        """

        completion = client.chat.completions.create(model=MODEL_NAME,
                                                    messages=[{"role": "system", "content": system_prompt},
                                                              {"role": "user", "content": user_prompt}])
        result = completion.choices[0].message.content
        print("智能决策完成。")
        return result
    except Exception as e:
        return f"调用最终决策API时发生错误: {e}"


def log_data_to_excel(user_data, lgbm_pred, formula_pred, final_pred_text):
    """
    功能：将所有输入和三个预测结果记录到Excel文件中。
    """
    print(f"\n正在记录数据到Excel文件: '{LOG_FILE_PATH}'...")
    try:
        # 兼容中英文标签提取
        final_amount_match = re.search(r'(?:【最终投喂量】|【Final Feeding Amount】):\s*(\d+\.?\d*)', final_pred_text)
        final_amount = float(final_amount_match.group(1)) if final_amount_match else '提取失败'

        headers_en = [
            'Date', 'month_day', 'system_id', 'average_water_temp', 'average_do', 'average_ph',
            'ammonia_nitrogen', 'nitrite_nitrogen', 'water_change_amount', 'water_change_rate',
            'age_in_days', 'weight', 'shrimp_loading_cap', 'water_body_capacity', 'survival_rate',
            'avg_daily_weight_gain', 'number_of_meals',
            'LGBM_Prediction_kg', 'Formula_Prediction_kg', 'Final_Predicted_Amount_kg',
            'Actual_Feeding_Amount_kg', 'Remarks'
        ]

        new_row_dict = {
            'Date': datetime.now().strftime('%Y%m%d'),
            'month_day': user_data.get('month_day'),
            'system_id': user_data.get('system_id'),
            'average_water_temp': user_data.get('average_water_temp'),
            'average_do': user_data.get('average_do'),
            'average_ph': user_data.get('average_ph'),
            'ammonia_nitrogen': user_data.get('ammonia_nitrogen'),
            'nitrite_nitrogen': user_data.get('nitrite_nitrogen'),
            'water_change_amount': user_data.get('water_change_amount'),
            'water_change_rate': user_data.get('water_change_rate'),
            'age_in_days': user_data.get('age_in_days'),
            'weight': user_data.get('weight'),
            'shrimp_loading_cap': user_data.get('shrimp_loading_cap'),
            'water_body_capacity': user_data.get('water_body_capacity'),
            'survival_rate': user_data.get('survival_rate'),
            'avg_daily_weight_gain': user_data.get('avg_daily_weight_gain'),
            'number_of_meals': user_data.get('number_of_meals'),
            'LGBM_Prediction_kg': lgbm_pred,
            'Formula_Prediction_kg': formula_pred,
            'Final_Predicted_Amount_kg': final_amount,
            'Actual_Feeding_Amount_kg': user_data.get('actual_feeding_amount'),
            'Remarks': user_data.get('remarks')
        }

        new_df = pd.DataFrame([new_row_dict])

        if not os.path.exists(LOG_FILE_PATH):
            df_to_save = new_df
        else:
            existing_df = pd.read_excel(LOG_FILE_PATH)
            df_to_save = pd.concat([existing_df, new_df], ignore_index=True)

        df_to_save.to_excel(LOG_FILE_PATH, index=False, engine='openpyxl', columns=headers_en)
        print("数据记录成功！")
    except PermissionError:
        print(f"\n错误：权限被拒绝！请关闭Excel文件 '{LOG_FILE_PATH}' 后重试。")
    except Exception as e:
        print(f"写入Excel时发生错误: {e}")