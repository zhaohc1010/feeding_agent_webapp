# -*- coding: utf-8 -*-
import os
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import HTTPException
from datetime import datetime
import re

# Firebase Admin SDK for database connection
import firebase_admin
from firebase_admin import credentials, firestore

from core_logic import (
    read_knowledge_base,
    predict_with_lightgbm,
    calculate_from_formulas,
    get_final_decision_with_remarks
)
from config import KNOWLEDGE_BASE_PATH

# --- Firebase Initialization ---
try:
    firebase_creds_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if not firebase_creds_json_str:
        print("FATAL ERROR: FIREBASE_CREDENTIALS_JSON environment variable not set.")
        db = None
    else:
        firebase_creds_dict = json.loads(firebase_creds_json_str)
        cred = credentials.Certificate(firebase_creds_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
except Exception as e:
    print(f"Failed to initialize Firebase: {e}")
    db = None
# --- End of Firebase Initialization ---


# 初始化 Flask 应用
app = Flask(__name__)

# 定义完整的列名顺序
FULL_COLUMNS = [
    'Date', 'month_day', 'system_id', 'average_water_temp', 'average_do', 'average_ph',
    'ammonia_nitrogen', 'nitrite_nitrogen', 'water_change_amount', 'water_change_rate',
    'age_in_days', 'weight', 'shrimp_loading_cap', 'water_body_capacity', 'survival_rate',
    'avg_daily_weight_gain', 'number_of_meals',
    'LGBM_Prediction_kg', 'Formula_Prediction_kg', 'Final_Predicted_Amount_kg',
    'Actual_Feeding_Amount_kg', 'Remarks'
]


# --- 全局错误处理器 ---
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    print(f"Unhandled Server Exception: {e}")
    import traceback
    traceback.print_exc()
    return jsonify(error=f"服务器发生意外错误: {str(e)}"), 500


# --- Helper Functions for Firestore ---

def get_system_and_logs_refs(system_id):
    """ Helper to get references and ensure parent document exists. """
    if not db or not system_id or not system_id.strip():
        return None, None
    system_id_stripped = system_id.strip()
    system_doc_ref = db.collection('systems').document(system_id_stripped)
    logs_collection_ref = system_doc_ref.collection('logs')
    return system_doc_ref, logs_collection_ref


def read_historical_data_from_firestore(system_id):
    _, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return "数据库未连接或系统ID无效。"
    try:
        docs = logs_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        records = [doc.to_dict() for doc in docs]
        if not records:
            return f"系统 {system_id} 尚无历史投喂记录。"

        df = pd.DataFrame(records)
        return df.to_string()

    except Exception as e:
        print(f"Error reading from Firestore for {system_id}: {e}")
        return f"读取历史记录时发生错误: {e}"


def log_data_to_firestore(system_id, user_data, lgbm_pred, formula_pred, final_pred_text):
    system_doc_ref, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        print("Firestore not initialized or invalid system_id, skipping log.")
        return
    try:
        final_amount_match = re.search(r'【最终投喂量】:\s*(\d+\.?\d*)', final_pred_text)
        final_amount = float(final_amount_match.group(1)) if final_amount_match else None

        new_log_data = {col: user_data.get(col) for col in FULL_COLUMNS if col in user_data}
        new_log_data.update({
            'Date': datetime.now().strftime('%Y%m%d'),
            'LGBM_Prediction_kg': lgbm_pred,
            'Formula_Prediction_kg': formula_pred,
            'Final_Predicted_Amount_kg': final_amount,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        system_doc_ref.set({'last_updated': firestore.SERVER_TIMESTAMP, 'system_id': system_id.strip()}, merge=True)

        logs_collection.add(new_log_data)
        print(f"Data successfully logged to Firestore for system {system_id}.")
    except Exception as e:
        print(f"Error logging to Firestore for {system_id}: {e}")


# --- Flask Routes ---

@app.route('/')
def index():
    if not db:
        return render_template('index.html', systems_data={}, error_message="数据库未连接，请检查服务器配置。")

    systems_data = {}
    try:
        system_docs = db.collection('systems').stream()
        for system in system_docs:
            system_id = system.id
            _, logs_collection = get_system_and_logs_refs(system_id)
            docs = logs_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
            records = [{'id': doc.id, **doc.to_dict()} for doc in docs]
            if records:
                systems_data[system_id] = records
    except Exception as e:
        print(f"Error fetching history data: {e}")
        return render_template('index.html', systems_data={}, error_message=f"加载历史记录失败: {e}")

    return render_template('index.html', systems_data=systems_data)


@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.json
    system_id = user_data.get('system_id')
    if not system_id or not system_id.strip():
        return jsonify({'error': '系统ID (System ID) 不能为空。'}), 400

    for key, value in user_data.items():
        if key not in ['system_id', 'month_day', 'remarks', 'actual_feeding_amount'] and (value is None or value == ''):
            return jsonify({'error': f'参数 {key} 不能为空。'}), 400
        if key not in ['system_id', 'month_day', 'remarks']:
            try:
                user_data[key] = float(value) if value is not None and value != '' else None
            except (ValueError, TypeError):
                pass

    lgbm_prediction = predict_with_lightgbm(user_data)
    formula_prediction_tuple = calculate_from_formulas(user_data)
    formula_prediction, _ = formula_prediction_tuple if formula_prediction_tuple else (None, "计算失败")

    final_recommendation = ""
    if lgbm_prediction is not None and formula_prediction is not None:
        if user_data.get('remarks') and user_data.get('remarks').strip():
            if not os.getenv("DASHSCOPE_API_KEY"):
                return jsonify({'error': '服务器配置错误：智能决策模块所需的API密钥未设置。'}), 500

            knowledge = read_knowledge_base()
            historical_data = read_historical_data_from_firestore(system_id)
            final_recommendation = get_final_decision_with_remarks(
                lgbm_prediction, formula_prediction_tuple, user_data,
                knowledge if knowledge else "知识库未找到或读取失败。", historical_data
            )
        else:
            final_recommendation = (f"根据LightGBM模型的计算，建议的投喂量为 {lgbm_prediction:.4f} 公斤。\n\n"
                                    f"【最终投喂量】: {lgbm_prediction:.4f}")
    else:
        final_recommendation = "一个或多个模型预测失败，无法生成最终建议。"

    if "失败" not in final_recommendation:
        log_data_to_firestore(system_id, user_data, lgbm_prediction, formula_prediction, final_recommendation)

    return jsonify({'recommendation': final_recommendation})


@app.route('/update_log_batch', methods=['POST'])
def update_log_batch():
    data = request.json
    system_id = data.get('system_id')
    changes = data.get('changes', {})

    _, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return jsonify({'error': '数据库未连接或系统ID无效'}), 500

    try:
        batch = db.batch()
        for doc_id, doc_changes in changes.items():
            doc_ref = logs_collection.document(doc_id)
            update_data = {}
            for column, value in doc_changes.items():
                if value and column not in ['Date', 'Remarks', 'system_id', 'month_day']:
                    try:
                        update_data[column] = float(value)
                    except (ValueError, TypeError):
                        update_data[column] = value
                else:
                    update_data[column] = value if value else None
            batch.update(doc_ref, update_data)
        batch.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'批量更新失败: {e}'}), 400


@app.route('/add_row', methods=['POST'])
def add_row():
    system_id = request.json.get('system_id')
    system_doc_ref, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return jsonify({'error': '数据库未连接或系统ID无效'}), 500

    try:
        system_doc_ref.set({'last_updated': firestore.SERVER_TIMESTAMP, 'system_id': system_id.strip()}, merge=True)

        new_row_data = {col: None for col in FULL_COLUMNS}
        new_row_data['timestamp'] = firestore.SERVER_TIMESTAMP
        new_row_data['system_id'] = system_id
        logs_collection.add(new_row_data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'新增行失败: {e}'}), 500


@app.route('/download_log/<system_id>')
def download_log(system_id):
    _, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return "数据库未连接或系统ID无效", 400
    try:
        docs = logs_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        records = [doc.to_dict() for doc in docs]
        if not records:
            return "日志为空，无可下载内容。", 404

        df = pd.DataFrame(records)
        df = df.reindex(columns=FULL_COLUMNS)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=f'Log_{system_id}')
        output.seek(0)

        return send_file(output, as_attachment=True, download_name=f'feeding_log_{system_id}.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        return f"下载失败: {e}", 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)

