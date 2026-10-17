# front.py

import os

import streamlit as st
import requests

# Set API_URL in the deployment environment to your public FastAPI URL.
# When running locally, the app falls back to the local backend.
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8091")
API_URL = API_URL.rstrip("/")

st.set_page_config("Food Nutrition Assistant")
st.title("🍎 Food Nutrition Assistant")

if "state_id" not in st.session_state:
    st.session_state.state_id = None
if "result" not in st.session_state:
    st.session_state.result = None
if "options" not in st.session_state:
    st.session_state.options = None

uploaded = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
text_input= st.text_input("Or enter food name")

if (uploaded or text_input) and st.button("Analyze"):
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)} if uploaded else None
    data = {"text": text_input} if text_input else None
    with st.spinner("Analyzing food..."):
        try:
            res = requests.post(
                f"{API_URL}/analyze-image",
                files=files,
                data=data,
                timeout=15
            )
            res.raise_for_status()
            data = res.json()

            if data["status"] == "need_confirmation":
                st.session_state.state_id = data["state_id"]
                st.session_state.options = data["options"]
                st.session_state.result = None
            elif data["status"] == "done":
                st.session_state.result = data
                st.session_state.options = None
            else:
                st.error(data.get("message", "Unable to analyze this food."))
        except requests.exceptions.Timeout:
            st.error("The nutrition backend is taking too long to respond. Please try again in a moment.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not connect to backend: {exc}")

# Ask user
if st.session_state.options:
    st.subheader("Which food is correct?")
    for opt in st.session_state.options:
        if st.button(opt):
            try:
                res = requests.post(
                    f"{API_URL}/confirm-food",
                    json={
                        "state_id": st.session_state.state_id,
                        "food": opt
                    },
                    timeout=15
                )
                res.raise_for_status()
                st.session_state.result = res.json()
                st.session_state.options = None
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not confirm food: {exc}")

# Final result
if st.session_state.result:
    r = st.session_state.result
    st.success(f"Food: {r['food']}")

    st.subheader("Nutrition")
    for k, v in r["nutrition"].items():
        st.write(f"**{k}**: {v}")

    st.subheader("Explanation & Recipe Ideas")
    st.write(r["explanation"])
