# -*- coding: utf-8 -*-
import os
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import HTTPException
from datetime import datetime
import re

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

from core_logic import (
    read_knowledge_base,
    predict_with_lightgbm,
    calculate_from_formulas,
    get_final_decision_with_remarks
)
from config import KNOWLEDGE_BASE_PATH
from translations import TRANSLATIONS

# --- Firebase Initialization ---
try:
    firebase_creds_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if not firebase_creds_json_str:
        print("FATAL ERROR: FIREBASE_CREDENTIALS_JSON not set.")
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
# --- End Firebase ---

app = Flask(__name__)

# Full columns list
FULL_COLUMNS = [
    'Date', 'month_day', 'system_id', 'average_water_temp', 'average_do', 'average_ph',
    'ammonia_nitrogen', 'nitrite_nitrogen', 'water_change_amount', 'water_change_rate',
    'age_in_days', 'weight', 'shrimp_loading_cap', 'water_body_capacity', 'survival_rate',
    'avg_daily_weight_gain', 'number_of_meals',
    'LGBM_Prediction_kg', 'Formula_Prediction_kg', 'Final_Predicted_Amount_kg',
    'Actual_Feeding_Amount_kg', 'Remarks'
]


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "error": e.description,
        })
        response.content_type = "application/json"
        return response
    print(f"Unhandled Server Exception: {e}")
    return jsonify(error=f"Server Error: {str(e)}"), 500


# --- Helpers ---
def get_system_and_logs_refs(system_id):
    if not db or not system_id or not system_id.strip():
        return None, None
    system_id_stripped = system_id.strip()
    system_doc_ref = db.collection('systems').document(system_id_stripped)
    logs_collection_ref = system_doc_ref.collection('logs')
    return system_doc_ref, logs_collection_ref


def read_historical_data_from_firestore(system_id):
    _, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return "No DB Connection or Invalid ID."
    try:
        docs = logs_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()
        records = [doc.to_dict() for doc in docs]
        if not records:
            return f"No history for system {system_id}."
        df = pd.DataFrame(records)
        return df.to_string()
    except Exception as e:
        print(f"Error reading Firestore: {e}")
        return f"Error reading history: {e}"


def log_data_to_firestore(system_id, user_data, lgbm_pred, formula_pred, final_pred_text):
    system_doc_ref, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return

    # Regex to support both languages
    final_amount_match = re.search(r'(?:【最终投喂量】|【Final Feeding Amount】):\s*(\d+\.?\d*)', final_pred_text)
    final_amount = float(final_amount_match.group(1)) if final_amount_match else None

    new_log_data = {
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
        'Remarks': user_data.get('remarks'),
        'timestamp': firestore.SERVER_TIMESTAMP
    }

    system_doc_ref.set({'last_updated': firestore.SERVER_TIMESTAMP, 'system_id': system_id.strip()}, merge=True)
    logs_collection.add(new_log_data)
    print(f"Data logged for {system_id}.")


# --- Routes ---

@app.route('/')
def index():

    lang = request.args.get('lang', 'en')
    if lang not in TRANSLATIONS:
        lang = 'en'
    t = TRANSLATIONS[lang]

    if not db:
        return render_template('index.html', systems_data={}, error_message="Database Disconnected.", t=t, lang=lang)

    systems_data = {}
    try:
        system_docs = db.collection('systems').order_by("last_updated", direction=firestore.Query.DESCENDING).stream()
        for system in system_docs:
            system_id = system.id
            _, logs_collection = get_system_and_logs_refs(system_id)
            docs = logs_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
            records = [{'id': doc.id, **doc.to_dict()} for doc in docs]
            if records:
                systems_data[system_id] = records
    except Exception as e:
        print(f"Error fetching data: {e}")
        return render_template('index.html', systems_data={}, error_message=str(e), t=t, lang=lang)

    return render_template('index.html', systems_data=systems_data, t=t, lang=lang)


@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.json


    lang = user_data.get('language', 'en')
    t = TRANSLATIONS[lang if lang in TRANSLATIONS else 'en']

    system_id = user_data.get('system_id')
    if not system_id or not system_id.strip():
        return jsonify({'error': 'System ID required.' if lang == 'en' else '需要系统ID。'}), 400

    # Data validation
    for key, value in user_data.items():
        if key not in ['system_id', 'month_day', 'remarks', 'language', 'actual_feeding_amount']:
            if value is None or value == '':
                msg = f'Missing: {key}' if lang == 'en' else f'缺少参数: {key}'
                return jsonify({'error': msg}), 400
            try:
                user_data[key] = float(value)
            except (ValueError, TypeError):
                pass

    lgbm_prediction = predict_with_lightgbm(user_data)
    formula_prediction_tuple = calculate_from_formulas(user_data)
    formula_prediction, _ = formula_prediction_tuple if formula_prediction_tuple else (None, "Calc Failed")

    final_recommendation = ""
    if lgbm_prediction is not None and formula_prediction is not None:
        if user_data.get('remarks') and user_data.get('remarks').strip():
            if not os.getenv("DASHSCOPE_API_KEY"):
                return jsonify({'error': 'API Key Missing.'}), 500

            knowledge = read_knowledge_base()
            historical_data = read_historical_data_from_firestore(system_id)

            final_recommendation = get_final_decision_with_remarks(
                lgbm_prediction, formula_prediction_tuple, user_data,
                knowledge if knowledge else "No KB.", historical_data,
                language=lang
            )
        else:
            prefix = t['lgbm_only_prefix']
            label = t['final_amount_label']
            final_recommendation = (f"{prefix} {lgbm_prediction:.4f} kg.\n\n"
                                    f"{label}: {lgbm_prediction:.4f}")
    else:
        final_recommendation = t['model_fail']

    if "失败" not in final_recommendation and "failed" not in final_recommendation.lower():
        log_data_to_firestore(system_id, user_data, lgbm_prediction, formula_prediction, final_recommendation)

    return jsonify({'recommendation': final_recommendation})


@app.route('/update_log_batch', methods=['POST'])
def update_log_batch():
    data = request.json
    system_id = data.get('system_id')
    changes = data.get('changes', {})
    _, logs_collection = get_system_and_logs_refs(system_id)

    if not logs_collection:
        return jsonify({'error': 'DB Error'}), 500

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


@app.route('/add_row', methods=['POST'])
def add_row():
    system_id = request.json.get('system_id')
    system_doc_ref, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection:
        return jsonify({'error': 'DB Error'}), 500

    system_doc_ref.set({'last_updated': firestore.SERVER_TIMESTAMP, 'system_id': system_id.strip()}, merge=True)
    new_row_data = {col: None for col in FULL_COLUMNS}
    new_row_data['timestamp'] = firestore.SERVER_TIMESTAMP
    new_row_data['system_id'] = system_id
    logs_collection.add(new_row_data)
    return jsonify({'success': True})


@app.route('/download_log/<system_id>')
def download_log(system_id):
    _, logs_collection = get_system_and_logs_refs(system_id)
    if not logs_collection: return "DB Error", 400
    docs = logs_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
    records = [doc.to_dict() for doc in docs]
    if not records: return "Empty", 404
    df = pd.DataFrame(records)
    export_columns = [col for col in FULL_COLUMNS if col in df.columns]
    df = df[export_columns]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=f'Log_{system_id}')
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f'log_{system_id}.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == '__main__':
    app.run(debug=True, port=5001)