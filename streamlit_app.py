import streamlit as st
import joblib
import os

ARTIFACT_PATH = "aiot_hw3/artifacts/spam_baseline.joblib"
REPORT_PATH = "aiot_hw3/artifacts/baseline_report.json"

st.title("Spam Email Classification Demo")

# 載入模型
if os.path.exists(ARTIFACT_PATH):
	pipe = joblib.load(ARTIFACT_PATH)
	st.success("模型已載入！")
else:
	st.warning(f"找不到模型檔案：{ARTIFACT_PATH}，請先執行訓練。")

# 載入指標
if os.path.exists(REPORT_PATH):
	import json
	with open(REPORT_PATH) as f:
		metrics = json.load(f)
	st.write("Baseline Metrics:", metrics)

# 預測介面
user_input = st.text_area("請輸入一段簡訊或郵件內容：")
if st.button("預測"):
	if user_input and 'pipe' in locals():
		pred = pipe.predict([user_input])[0]
		st.write(f"預測結果：{'Spam' if pred == 1 else 'Ham'}")
	else:
		st.warning("請先輸入內容並確認模型已載入。")
