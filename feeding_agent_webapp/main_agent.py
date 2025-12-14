# -*- coding: utf-8 -*-

import os


from core_logic import (
    read_knowledge_base,
    read_historical_data,
    predict_with_lightgbm,
    calculate_from_formulas,  
    get_final_decision_with_remarks,
    log_data_to_excel
)


def get_user_input():

    print("=" * 50)
    print("欢迎使用智能投喂量预测智能体")
    print("Welcome to the Smart Feeding Amount Prediction Agent")
    print("=" * 50)
    print("请输入当前的养殖参数 / Please enter the current farming parameters:\n")

    params_prompts = {
        "month_day": "请输入月日 (Enter Month-Day, e.g., 704 for July 4th): ",
        "system_id": "请输入系统ID (Enter System ID): ",
        "average_water_temp": "请输入平均水温 (Enter Average Water Temperature, °C): ",
        "average_do": "请输入平均溶解氧 (Enter Average Dissolved Oxygen, mg/L): ",
        "average_ph": "请输入平均pH值 (Enter Average pH): ",
        "ammonia_nitrogen": "请输入氨氮浓度 (Enter Ammonia Nitrogen, mg/L): ",
        "nitrite_nitrogen": "请输入亚硝酸盐浓度 (Enter Nitrite Nitrogen, mg/L): ",
        "water_change_amount": "请输入换水量 (Enter Water Change Amount, m³): ",
        "water_change_rate": "请输入换水率 (Enter Water Change Rate, %): ",
        "age_in_days": "请输入养殖天数 (Enter Age in Days): ",
        "weight": "请输入单只虾平均体重 (Enter Average Shrimp Weight, g): ",
        "shrimp_loading_cap": "请输入存塘量 (Enter Shrimp Loading Capacity, Jin/m³): ",
        "water_body_capacity": "请输入水体容量 (Enter Water Body Capacity, m³): ",
        "survival_rate": "请输入成活率预估 (Enter Estimated Survival Rate, e.g., 0.99): ",
        "avg_daily_weight_gain": "请输入日均增重 (Enter Avg. Daily Weight Gain, g): ",
        "number_of_meals": "请输入日投喂餐数 (Enter Number of Meals per Day): ",
        "actual_feeding_amount": "请输入实际投喂量 (Enter Actual Feeding Amount, kg) [可选/Optional]: ",
        "remarks": "请输入备注 (Enter Remarks) [可选/Optional]: "
    }

    user_data = {}
    for key, prompt in params_prompts.items():
        while True:
            try:
                value = input(prompt)
                if key == 'actual_feeding_amount':
                    user_data[key] = float(value) if value else ""
                    break
                elif key == 'remarks':
                    user_data[key] = value
                    break
                elif key not in ["system_id", "month_day"]:
                    user_data[key] = float(value)
                    break
                else:
                    user_data[key] = value
                    break
            except ValueError:
                print("输入无效，请输入一个数字。 / Invalid input, please enter a number.")
            except Exception as e:
                print(f"发生未知错误: {e} / An unknown error occurred: {e}")

    print("\n信息收集完毕，感谢您的输入！")
    return user_data



def run_agent():

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("=" * 50, "\n错误：环境变量 'DASHSCOPE_API_KEY' 未设置。\n", "=" * 50)
        return

    # 1
    user_data = get_user_input()

    # 2. LightGBM
    lgbm_prediction = predict_with_lightgbm(user_data)

    # 3.
    formula_prediction_tuple = calculate_from_formulas(user_data)
    formula_prediction, formula_explanation = formula_prediction_tuple if formula_prediction_tuple is not None else (
    None, "计算失败")


    print("\n" + "-" * 50)
    print("内置公式计算详情 / Formula Calculation Details")
    print("-" * 50)
    print(formula_explanation)
    print("-" * 50)


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


    print("\n" + "=" * 50)
    print("最终投喂建议 / Final Feeding Recommendation")
    print("=" * 50)
    print(final_recommendation)


    if "失败" not in final_recommendation:
        log_data_to_excel(user_data, lgbm_prediction, formula_prediction, final_recommendation)


if __name__ == "__main__":

    run_agent()
