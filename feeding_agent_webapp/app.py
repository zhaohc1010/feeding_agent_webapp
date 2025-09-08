# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, jsonify
from core_logic import (
    read_knowledge_base,
    read_historical_data,
    predict_with_lightgbm,
    calculate_from_formulas,
    get_final_decision_with_remarks,
    log_data_to_excel
)

# 初始化 Flask 应用
app = Flask(__name__)

# 定义根路由，用于提供前端页面
@app.route('/')
def index():
    """渲染并返回主页面 index.html"""
    return render_template('index.html')

# 定义预测API接口，只接受POST请求
@app.route('/predict', methods=['POST'])
def predict():
    """接收前端发来的数据，执行核心逻辑，并返回预测结果"""
    try:
        # 从请求中获取JSON格式的用户数据
        user_data = request.json
        print("接收到前端数据:", user_data)

        # --- 数据类型转换 ---
        # 从前端JSON收到的数据都是字符串，需要将数值型字段转换为浮点数
        for key in user_data:
            # 这些字段保持字符串或可以为空
            if key in ['system_id', 'month_day', 'remarks', 'actual_feeding_amount']:
                continue
            # 其他字段必须是有效数字
            if user_data[key]:
                user_data[key] = float(user_data[key])
            else:
                # 如果必填项为空，返回错误
                return jsonify({'error': f'参数 {key} 不能为空。'}), 400

        # 处理可选字段
        if user_data.get('actual_feeding_amount') == '':
            user_data['actual_feeding_amount'] = None
        else:
            user_data['actual_feeding_amount'] = float(user_data.get('actual_feeding_amount', 0))

        # --- 执行核心逻辑 ---
        lgbm_prediction = predict_with_lightgbm(user_data)
        formula_prediction_tuple = calculate_from_formulas(user_data)
        formula_prediction, formula_explanation = formula_prediction_tuple if formula_prediction_tuple else (None, "计算失败")

        final_recommendation = ""
        if lgbm_prediction is not None and formula_prediction is not None:
            if user_data.get('remarks'):
                knowledge = read_knowledge_base()
                historical_data = read_historical_data()
                final_recommendation = get_final_decision_with_remarks(
                    lgbm_prediction,
                    formula_prediction_tuple,
                    user_data,
                    knowledge if knowledge else "知识库未找到或读取失败。",
                    historical_data
                )
            else:
                final_recommendation = (
                    f"根据LightGBM模型的计算，建议的投喂量为 {lgbm_prediction:.4f} 公斤。\n"
                    f"由于未提供备注信息，系统默认采纳此数据驱动模型的预测结果。\n\n"
                    f"【最终投喂量】: {lgbm_prediction:.4f}"
                )
        else:
            final_recommendation = "一个或多个模型预测失败，无法生成最终建议。"

        # 如果预测成功，则记录日志
        if "失败" not in final_recommendation:
            # 确保日志文件夹存在
            os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), 'data', 'placeholder.txt')), exist_ok=True)
            log_data_to_excel(user_data, lgbm_prediction, formula_prediction, final_recommendation)

        # 以JSON格式返回最终结果
        return jsonify({'recommendation': final_recommendation})

    except Exception as e:
        # 捕获任何可能的错误，并返回一个清晰的错误信息
        print(f"服务器发生错误: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    # 这行代码只在本地测试时运行。
    # 在云端部署时，会使用一个专业的服务器（如 Gunicorn）来启动应用。
    app.run(debug=True, port=5001)