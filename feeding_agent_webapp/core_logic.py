# -*- coding: utf-8 -*-


import os
import re
import warnings
import docx
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI
from sklearn.exceptions import InconsistentVersionWarning
from config import LOG_FILE_PATH, MODEL_NAME, BASE_URL, LGBM_MODEL_PATH, KNOWLEDGE_BASE_PATH

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def read_knowledge_base():
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return None
    try:
        document = docx.Document(KNOWLEDGE_BASE_PATH)
        return '\n'.join([para.text for para in document.paragraphs])
    except Exception:
        return None


def read_historical_data():
    if not os.path.exists(LOG_FILE_PATH):
        return "尚无记录。"
    try:
        df = pd.read_excel(LOG_FILE_PATH)
        return df.tail(10).to_string() if not df.empty else "记录为空。"
    except Exception:
        return "读取失败。"


def predict_with_lightgbm(user_data):

    print("\n[LightGBM] 启动预测...")
    try:
        loaded = joblib.load(LGBM_MODEL_PATH)
        model = loaded['model']
        input_scaler = loaded['input_scaler']
        output_scaler = loaded['output_scaler']
    except Exception as e:
        print(f"[LightGBM Error] 模型加载失败: {e}")
        return None

    feature_keys = [
        "month_day", "system_id", "average_water_temp", "average_do", "average_ph",
        "ammonia_nitrogen", "nitrite_nitrogen", "water_change_amount", "water_change_rate",
        "age_in_days", "weight", "shrimp_loading_cap", "water_body_capacity",
        "survival_rate", "avg_daily_weight_gain", "number_of_meals"
    ]

    missing = [k for k in feature_keys if k not in user_data]
    if missing:
        print(f"[LightGBM Error] 缺失字段: {missing}")
        return None

    input_df = pd.DataFrame([user_data])[feature_keys]

    try:
        input_data_scaled = input_scaler.transform(input_df)
        prediction_scaled = model.predict(input_data_scaled)
        final_prediction = output_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
        result = final_prediction[0][0]
        print(f"[LightGBM] 结果: {result:.4f} kg")
        return result
    except Exception as e:
        print(f"[LightGBM Error] 计算异常: {e}")
        return None


def calculate_from_formulas(user_data):

    print("\n[Formula] 启动计算...")
    try:
        w = float(user_data.get('weight', 0))
        cap = float(user_data.get('shrimp_loading_cap', 0))
        vol = float(user_data.get('water_body_capacity', 0))
    except:
        return None, "参数需为数字"

    if w <= 0: return None, "体重需大于0"

    count = cap * 500 * vol / w


    coeffs = None
    if w < 0.8:
        coeffs = (0.144, 0.162)
    elif w < 1.5:
        coeffs = (0.129, 0.134)
    elif w < 2.4:
        coeffs = (0.097, 0.099)
    elif w < 3.6:
        coeffs = (0.084, 0.085)
    elif w < 5.21:
        coeffs = (0.065, 0.076)
    elif w < 7.1:
        coeffs = (0.061, 0.072)
    elif w < 9.3:
        coeffs = (0.058, 0.062)
    elif w < 11.63:
        coeffs = (0.045, 0.052)
    elif w < 13.51:
        coeffs = (0.040, 0.044)
    elif w < 15.9:
        coeffs = (0.032, 0.040)
    elif w < 17.2:
        coeffs = (0.032, 0.037)
    elif w < 18.6:
        coeffs = (0.030, 0.030)
    elif w < 19.7:
        coeffs = (0.025, 0.025)
    else:
        coeffs = (0.025, 0.025)

    low = w * coeffs[0] * count * 0.001
    high = w * coeffs[1] * count * 0.001
    avg = (low + high) / 2

    print(f"[Formula] 结果: {avg:.4f} kg")
    return avg, f"体重{w}g, 系数{coeffs}"


def get_final_decision_with_remarks(lgbm_pred, formula_pred_tuple, user_data, knowledge_content, historical_data_str,
                                    language='en'):

    print(f"\n[AI Decision] 启动决策 (Lang: {language})...")
    formula_pred, formula_explanation = formula_pred_tuple

    l_val = f"{lgbm_pred:.4f}" if lgbm_pred is not None else "Failed"
    f_val = f"{formula_pred:.4f}" if formula_pred is not None else "Failed"

    try:
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=BASE_URL)


        if language == 'en':
            system_prompt = """
            You are a Chief Expert in shrimp farming. Your task is to be the final decision maker.

            Your decision process must STRICTLY follow these steps (Execute these in your mind, but output a CONCISE summary):
            1. Prioritize analyzing key information in the [User Remarks].
            2. Combine [Knowledge Base] and [Historical Records] to understand the impact of the remarks.
            3. Synthesize all the information, evaluate which of the two predicted values is more reasonable, or propose a better adjustment value..
            4. Make a final choice.
            
            Generate a decision report (about 500 words).
            At the end, you MUST strictly follow the format '【Final Feeding Amount】: XX.XX' on a new line.
            """
            user_intro = "Please analyze based on the info above and provide a concise decision in English."
            labels = ["LightGBM Prediction", "Formula Result", "Knowledge Base", "History Records", "User Remarks"]
        else:
            system_prompt = """
            你是一位经验丰富的对虾养殖首席专家，任务是作为最终决策者。

            你的决策流程必须严格执行以下步骤（请在后台深度思考，但输出需精简）：
            1. 优先分析【用户备注】中的关键信息。
            2. 结合【知识库全文】和【历史投喂记录】，理解备注信息对投喂量的影响。
            3. 综合所有信息，评估两个预测值哪个更合理，或提出一个更优的调整值。
            4. 做出最终选择。
            
            生成一份决策报告（约600字左右）
            在回答的最后，必须另起一行，并严格按照 '【最终投喂量】: XX.XX' 的格式给出你最终采纳的投喂量数值。
            """
            user_intro = "请根据以上信息进行专家决策"
            labels = ["LightGBM模型预测值", "内置公式计算结果", "知识库全文", "历史投喂记录", "用户备注"]

        user_prompt = f"""
        【{labels[0]}】: {l_val} kg
        【{labels[1]}】: {f_val} kg (依据: {formula_explanation})
        【{labels[4]}】: {user_data.get('remarks')}

        【{labels[3]}】 (最近10条):
        {historical_data_str}

        【{labels[2]}】 (摘要):
        {knowledge_content[:800] if knowledge_content else 'N/A'}...

        {user_intro}
        """

        completion = client.chat.completions.create(model=MODEL_NAME,
                                                    messages=[{"role": "system", "content": system_prompt},
                                                              {"role": "user", "content": user_prompt}])
        result = completion.choices[0].message.content
        print("[AI Decision] 完成。")
        return result
    except Exception as e:
        return f"AI API Error: {e}"


def log_data_to_excel(user_data, lgbm_pred, formula_pred, final_pred_text):
    print(f"\n记录 Excel: '{LOG_FILE_PATH}'...")
    try:
        match = re.search(r'(?:【最终投喂量】|【Final Feeding Amount】):\s*(\d+\.?\d*)', final_pred_text)
        final_amt = float(match.group(1)) if match else None

        row = {
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
            'Final_Predicted_Amount_kg': final_amt,
            'Actual_Feeding_Amount_kg': user_data.get('actual_feeding_amount'),
            'Remarks': user_data.get('remarks')
        }


        headers = list(row.keys())
        new_df = pd.DataFrame([row])

        if not os.path.exists(LOG_FILE_PATH):
            df_to_save = new_df
        else:
            existing = pd.read_excel(LOG_FILE_PATH)
            df_to_save = pd.concat([existing, new_df], ignore_index=True)

        df_to_save.to_excel(LOG_FILE_PATH, index=False, engine='openpyxl')
        print("Excel 记录成功。")
    except Exception as e:

        print(f"Excel 记录失败: {e}")
