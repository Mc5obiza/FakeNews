import datetime as dt
import os

import requests
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.title("Fake News Detector")
st.caption("Classify a news article as real or fake using your FastAPI backend model.")

with st.sidebar:
    st.subheader("Backend")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).rstrip("/")

    if st.button("Check API health"):
        try:
            health_response = requests.get(f"{api_url}/health", timeout=10)
            if health_response.ok:
                st.success("API is healthy")
            else:
                st.error(f"Health check failed: {health_response.status_code}")
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")

with st.form("predict_form"):
    title = st.text_input("Title")
    text = st.text_area("Article text", height=220)
    subject = st.text_input("Subject", value="politics")
    date = st.date_input("Publication date", value=dt.date.today())

    submitted = st.form_submit_button("Predict")

if submitted:
    if not title.strip() or not text.strip():
        st.warning("Title and article text are required.")
    else:
        payload = {
            "title": title.strip(),
            "text": text.strip(),
            "subject": subject.strip(),
            "date": date.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=20)
            if response.ok:
                result = response.json()
                prediction = result["prediction"]
                probability_real = float(result["probability_real"])
                probability_fake = float(result["probability_fake"])

                st.subheader("Prediction Result")
                if prediction == "real":
                    st.success("Prediction: REAL")
                else:
                    st.error("Prediction: FAKE")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probability real", f"{probability_real:.2%}")
                with col2:
                    st.metric("Probability fake", f"{probability_fake:.2%}")
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Prediction failed ({response.status_code}): {detail}")
        except requests.RequestException as exc:
            st.error(f"Request error: {exc}")
