# -*- coding: utf-8 -*-
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from core_logic import (
    read_knowledge_base,
    read_historical_data,
    predict_with_lightgbm,
    calculate_from_formulas,
    get_final_decision_with_remarks,
    log_data_to_excel
)
from config import LOG_FILE_PATH

# 初始化 Flask 应用
app = Flask(__name__)


def get_history_table_html():
    """
    辅助函数：读取日志文件并生成HTML表格。
    """
    if not os.path.exists(LOG_FILE_PATH):
        return "<p class='text-center text-gray-500'>暂无历史记录</p>"

    try:
        df = pd.read_excel(LOG_FILE_PATH)
        if df.empty:
            return "<p class='text-center text-gray-500'>暂无历史记录</p>"

        all_records = df.iloc[::-1]
        display_columns = [
            'Date', 'age_in_days', 'weight', 'average_water_temp', 'average_do',
            'LGBM_Prediction_kg', 'Final_Predicted_Amount_kg',
            'Actual_Feeding_Amount_kg', 'Remarks'
        ]
        existing_columns = [col for col in display_columns if col in all_records.columns]

        history_table_html = all_records[existing_columns].to_html(
            classes='min-w-full divide-y divide-gray-200', header=True,
            index=False, na_rep='-', border=0
        )
        history_table_html = history_table_html.replace('<thead>', '<thead class="bg-gray-50">')
        history_table_html = history_table_html.replace('<tbody>', '<tbody class="bg-white divide-y divide-gray-200">')
        history_table_html = history_table_html.replace('<th>',
                                                        '<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">')
        history_table_html = history_table_html.replace('<td>',
                                                        '<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-700">')
        return history_table_html

    except Exception as e:
        print(f"读取日志文件时出错: {e}")
        return "<p class='text-center text-red-500'>读取历史记录失败</p>"


@app.route('/')
def index():
    """
    渲染主页, 加载全部历史记录。
    """
    return render_template('index.html', history_table=get_history_table_html())


@app.route('/predict', methods=['POST'])
def predict():
    """
    接收前端数据，执行逻辑，并返回预测结果和更新后的历史记录。
    """
    try:
        user_data = request.json
        # ... (数据类型转换逻辑保持不变) ...
        for key in user_data:
            if key in ['system_id', 'month_day', 'remarks', 'actual_feeding_amount']: continue
            if user_data[key]:
                user_data[key] = float(user_data[key])
            else:
                return jsonify({'error': f'参数 {key} 不能为空。'}), 400

        if user_data.get('actual_feeding_amount') == '':
            user_data['actual_feeding_amount'] = None
        else:
            user_data['actual_feeding_amount'] = float(user_data.get('actual_feeding_amount', 0))

        # ... (模型调用和决策逻辑保持不变) ...
        lgbm_prediction = predict_with_lightgbm(user_data)
        formula_prediction_tuple = calculate_from_formulas(user_data)
        formula_prediction, _ = formula_prediction_tuple if formula_prediction_tuple else (None, "计算失败")
        final_recommendation = ""
        if lgbm_prediction is not None and formula_prediction is not None:
            if user_data.get('remarks'):
                knowledge = read_knowledge_base()
                historical_data = read_historical_data()
                final_recommendation = get_final_decision_with_remarks(
                    lgbm_prediction, formula_prediction_tuple, user_data,
                    knowledge if knowledge else "知识库未找到或读取失败。", historical_data
                )
            else:
                final_recommendation = (
                    f"根据LightGBM模型的计算，建议的投喂量为 {lgbm_prediction:.4f} 公斤。\n"
                    f"由于未提供备注信息，系统默认采纳此数据驱动模型的预测结果。\n\n"
                    f"【最终投喂量】: {lgbm_prediction:.4f}"
                )
        else:
            final_recommendation = "一个或多个模型预测失败，无法生成最终建议。"

        response_data = {'recommendation': final_recommendation}

        if "失败" not in final_recommendation:
            os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
            log_data_to_excel(user_data, lgbm_prediction, formula_prediction, final_recommendation)
            # 在记录成功后，获取更新后的历史记录HTML
            response_data['history_table'] = get_history_table_html()

        return jsonify(response_data)

    except Exception as e:
        print(f"服务器发生错误: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/download_log')
def download_log():
    """
    提供Excel日志文件的下载功能 (此函数逻辑不变)。
    """
    try:
        if not os.path.exists(LOG_FILE_PATH):
            return "日志文件尚未生成。", 404
        return send_file(LOG_FILE_PATH, as_attachment=True, download_name='feeding_log.xlsx')
    except Exception as e:
        print(f"下载文件时出错: {e}")
        return "下载文件时发生服务器内部错误。", 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)

