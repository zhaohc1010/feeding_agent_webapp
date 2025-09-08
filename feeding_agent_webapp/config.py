# -*- coding: utf-8 -*-
"""
配置模块
将所有可配置的路径和参数集中在此处，方便管理。
"""
import os

# 获取项目的根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 知识库文件路径 (使用相对路径)
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "Feeding rules.docx")

# 2. 日志文件路径 (使用相对路径, 存放在新建的data文件夹中)
LOG_FILE_PATH = os.path.join(BASE_DIR, "data", "feeding_log.xlsx")

# 3. AI模型配置
MODEL_NAME = "qwen-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 4. LightGBM 模型路径配置 (使用相对路径)
LGBM_MODEL_PATH = os.path.join(BASE_DIR, "best_lightgbm_model_ALLD4.joblib")