# -*- coding: utf-8 -*-
"""
核心逻辑模块
"""

import os
import re
import warnings
import docx
import joblib
import numpy as np
import pandas as pd
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


def predict_with_lightgbm(user_data):
    try:
        loaded_artifacts = joblib.load(LGBM_MODEL_PATH)
        model = loaded_artifacts['model']
        input_scaler = loaded_artifacts['input_scaler']
        output_scaler = loaded_artifacts['output_scaler']
    except FileNotFoundError:
        return None

    feature_keys = [
        "month_day", "system_id", "average_water_temp", "average_do", "average_ph",
        "ammonia_nitrogen", "nitrite_nitrogen", "water_change_amount", "water_change_rate",
        "age_in_days", "weight", "shrimp_loading_cap", "water_body_capacity",
        "survival_rate", "avg_daily_weight_gain", "number_of_meals"
    ]

    input_df = pd.DataFrame([user_data])[feature_keys]
    input_data_numpy = input_df.values.astype(np.float32)
    input_data_scaled = input_scaler.transform(input_data_numpy)
    prediction_scaled = model.predict(input_data_scaled)
    final_prediction = output_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    return final_prediction[0][0]


def calculate_from_formulas(user_data):
    weight = user_data.get('weight')
    shrimp_loading_cap = user_data.get('shrimp_loading_cap')
    water_body_capacity = user_data.get('water_body_capacity')

    if not all(isinstance(x, (int, float)) for x in [weight, shrimp_loading_cap, water_body_capacity]):
        return None, "Missing parameters."

    try:
        if weight == 0: raise ZeroDivisionError
        estimated_shrimp_count = shrimp_loading_cap * 500 * water_body_capacity / weight
    except ZeroDivisionError:
        return None, "Weight is 0."

    coeffs = None
    # 简化判断逻辑，保持原样即可，这里省略部分代码以节省篇幅，实际应保留原有的if-elif链
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
        return None, f"Weight {weight}g out of range."

    low = weight * coeffs[0] * estimated_shrimp_count * 0.001
    high = weight * coeffs[1] * estimated_shrimp_count * 0.001
    avg = (low + high) / 2
    return avg, f"Range: {low:.2f}-{high:.2f} kg"


def get_final_decision_with_remarks(lgbm_pred, formula_pred_tuple, user_data, knowledge_content, historical_data_str,
                                    language='en'):
    formula_pred, formula_explanation = formula_pred_tuple
    try:
        client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=BASE_URL)

        if language == 'en':
            system_prompt = """
            You are a shrimp farming expert.
            Task: Decide the final feeding amount based on Model Prediction, Formula, History, and Remarks.

            IMPORTANT:
            1. Be CONCISE. Do not show long reasoning steps.
            2. Summarize key factors from Remarks/History in 2-3 sentences.
            3. State which value you trust more and why (briefly).
            4. End with '【Final Feeding Amount】: XX.XX' on a new line.
            """
            user_intro = "Answer in English. Keep it short."
        else:
            system_prompt = """
            你是对虾养殖专家。
            任务：结合模型预测、公式、历史和备注决定最终投喂量。

            重要要求：
            1. 必须简练。不要输出冗长的思考过程。
            2. 用2-3句话总结备注和历史记录中的关键点。
            3. 简述你采纳某个数值的理由。
            4. 结尾必须另起一行输出：'【最终投喂量】: XX.XX'。
            """
            user_intro = "请用中文回答，保持简短。"

        user_prompt = f"""
        [LGBM]: {lgbm_pred:.4f} kg
        [Formula]: {formula_pred:.4f} kg ({formula_explanation})
        [Remarks]: {user_data.get('remarks')}
        [History]: {historical_data_str}
        [Current]: {user_data}
        [Knowledge]: {knowledge_content[:500]}...

        {user_intro}
        """

        completion = client.chat.completions.create(model=MODEL_NAME,
                                                    messages=[{"role": "system", "content": system_prompt},
                                                              {"role": "user", "content": user_prompt}])
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"