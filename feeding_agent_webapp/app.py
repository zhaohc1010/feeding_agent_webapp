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
    辅助函数：读取日志文件并生成可编辑的HTML表格。
    """
    if not os.path.exists(LOG_FILE_PATH):
        return "<p class='text-center text-gray-500'>暂无历史记录</p>"

    try:
        df = pd.read_excel(LOG_FILE_PATH)
        if df.empty:
            return "<p class='text-center text-gray-500'>暂无历史记录</p>"

        # 为了让最新的记录在最上面，我们反转DataFrame的行
        df_reversed = df.iloc[::-1]

        display_columns = [
            'Date', 'age_in_days', 'weight', 'average_water_temp', 'average_do',
            'LGBM_Prediction_kg', 'Final_Predicted_Amount_kg',
            'Actual_Feeding_Amount_kg', 'Remarks'
        ]

        # 确保我们只显示存在的列
        existing_columns = [col for col in display_columns if col in df_reversed.columns]

        # 手动构建HTML表格以添加可编辑属性
        html = '<table class="min-w-full divide-y divide-gray-200">'
        html += '<thead class="bg-gray-50"><tr>'
        for col in existing_columns:
            html += f'<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{col}</th>'
        html += '</tr></thead>'
        html += '<tbody class="bg-white divide-y divide-gray-200">'

        for index, row in df_reversed.iterrows():
            # 使用原始DataFrame的索引作为行的唯一标识
            original_index = index
            html += f'<tr data-original-index="{original_index}">'
            for col in existing_columns:
                cell_value = row[col]
                # 对NaN值进行处理，显示为空字符串
                if pd.isna(cell_value):
                    cell_value = ''
                html += (f'<td contenteditable="true" data-column="{col}" '
                         f'class="px-6 py-4 whitespace-nowrap text-sm text-gray-700 outline-none focus:bg-yellow-100">'
                         f'{cell_value}</td>')
            html += '</tr>'

        html += '</tbody></table>'
        return html

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
    接收前端数据，执行逻辑，并根据是否有备注决定决策流程。
    """
    try:
        user_data = request.json
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

        lgbm_prediction = predict_with_lightgbm(user_data)
        formula_prediction_tuple = calculate_from_formulas(user_data)
        formula_prediction, _ = formula_prediction_tuple if formula_prediction_tuple else (None, "计算失败")
        final_recommendation = ""

        if lgbm_prediction is not None and formula_prediction is not None:
            if user_data.get('remarks') and user_data.get('remarks').strip():
                knowledge = read_knowledge_base()
                historical_data = read_historical_data()
                final_recommendation = get_final_decision_with_remarks(
                    lgbm_prediction, formula_prediction_tuple, user_data,
                    knowledge if knowledge else "知识库未找到或读取失败。", historical_data
                )
            else:
                final_recommendation = (
                    f"根据LightGBM模型的计算，建议的投喂量为 {lgbm_prediction:.4f} 公斤。\n\n"
                    f"由于未提供备注信息，系统默认采纳此数据驱动模型的预测结果。\n\n"
                    f"【最终投喂量】: {lgbm_prediction:.4f}"
                )
        else:
            final_recommendation = "一个或多个模型预测失败，无法生成最终建议。"

        response_data = {'recommendation': final_recommendation}

        if "失败" not in final_recommendation:
            os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
            log_data_to_excel(user_data, lgbm_prediction, formula_prediction, final_recommendation)

        return jsonify(response_data)

    except Exception as e:
        print(f"服务器发生错误: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/update_log', methods=['POST'])
def update_log():
    """
    接收前端发来的修改，更新Excel文件。
    """
    try:
        data = request.json
        index = int(data['index'])
        column = data['column']
        value = data['value']

        if not os.path.exists(LOG_FILE_PATH):
            return jsonify({'error': 'Log file not found.'}), 404

        df = pd.read_excel(LOG_FILE_PATH)

        # 更新DataFrame中对应位置的值
        # 我们需要尝试将值转换为正确的类型
        try:
            # 如果原始列是数字类型，尝试转换新值
            if pd.api.types.is_numeric_dtype(df[column]):
                value = float(value)
            df.loc[index, column] = value
        except (ValueError, KeyError) as e:
            # 如果转换失败或列名错误，也视为客户端错误
            return jsonify({'error': f'Invalid value or column: {e}'}), 400

        # 保存回Excel文件
        df.to_excel(LOG_FILE_PATH, index=False)

        return jsonify({'success': True, 'message': f'Row {index}, Column {column} updated.'})

    except Exception as e:
        print(f"更新日志时发生错误: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/download_log')
def download_log():
    """
    提供Excel日志文件的下载功能。
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

