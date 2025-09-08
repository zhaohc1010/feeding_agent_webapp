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


@app.route('/')
def index():
    """
    渲染主页，加载仪表盘统计数据和历史记录。
    """
    dashboard_stats = {
        'total_predictions': 0,
        'avg_temp': 'N/A',
        'avg_do': 'N/A',
        'avg_final_kg': 'N/A'
    }
    history_table_html = "<h3>暂无历史记录</h3>"

    if os.path.exists(LOG_FILE_PATH):
        try:
            df = pd.read_excel(LOG_FILE_PATH)
            if not df.empty:
                # 计算仪表盘统计数据
                dashboard_stats['total_predictions'] = len(df)

                # 确保列存在且为数字类型再计算
                if 'average_water_temp' in df.columns and pd.to_numeric(df['average_water_temp'],
                                                                        errors='coerce').notna().any():
                    dashboard_stats['avg_temp'] = f"{df['average_water_temp'].astype(float).mean():.2f}°C"

                if 'average_do' in df.columns and pd.to_numeric(df['average_do'], errors='coerce').notna().any():
                    dashboard_stats['avg_do'] = f"{df['average_do'].astype(float).mean():.2f} mg/L"

                if 'Final_Predicted_Amount_kg' in df.columns and pd.to_numeric(df['Final_Predicted_Amount_kg'],
                                                                               errors='coerce').notna().any():
                    dashboard_stats['avg_final_kg'] = f"{df['Final_Predicted_Amount_kg'].astype(float).mean():.2f} kg"

                # 准备历史记录表格
                latest_records = df.tail(20)
                display_columns = [
                    'Date', 'age_in_days', 'weight', 'average_water_temp', 'average_do',
                    'LGBM_Prediction_kg', 'Formula_Prediction_kg', 'Final_Predicted_Amount_kg',
                    'Actual_Feeding_Amount_kg', 'Remarks'
                ]
                existing_columns = [col for col in display_columns if col in latest_records.columns]
                history_table_html = latest_records[existing_columns].to_html(
                    classes='history-table',
                    index=False,
                    na_rep='-'
                )
        except Exception as e:
            print(f"读取日志文件时出错: {e}")
            history_table_html = "<h3>读取历史记录失败</h3>"

    # 将仪表盘数据和历史记录表格都传递给前端
    return render_template('index.html', stats=dashboard_stats, history_table=history_table_html)


@app.route('/predict', methods=['POST'])
def predict():
    """
    接收前端发来的数据，执行核心逻辑，并返回预测结果 (此函数逻辑不变)。
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

        if "失败" not in final_recommendation:
            os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
            log_data_to_excel(user_data, lgbm_prediction, formula_prediction, final_recommendation)

        return jsonify({'recommendation': final_recommendation})
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
        return send_file(LOG_FILE_PATH, as_attachment=True)
    except Exception as e:
        print(f"下载文件时出错: {e}")
        return "下载文件时发生服务器内部错误。", 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)

