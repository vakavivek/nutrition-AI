import os
import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="NutriLens AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR THE SLEEK INSPIRATION LOOK ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0b0f17;
        color: #f9fafb;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Buttons styled with the neon lime theme */
    .stButton>button {
        background-color: #bbf246;
        color: #0b0f17;
        border-radius: 8px;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #a3e635;
        color: #000000;
        box-shadow: 0 0 15px rgba(187, 242, 70, 0.4);
    }
    
    /* Input fields and boxes */
    .stTextInput>div>div>input, .stFileUploader {
        background-color: #111827;
        color: #f9fafb;
        border-radius: 8px;
        border: 1px solid #374151;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 6px;
        color: #9ca3af;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #bbf246 !important;
        color: #0b0f17 !important;
    }
</style>
""", unsafe_allow_html=True)

# Set API URL
try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8091")
API_URL = API_URL.rstrip("/")

# Initialize session state
if "state_id" not in st.session_state:
    st.session_state.state_id = None
if "result" not in st.session_state:
    st.session_state.result = None
if "options" not in st.session_state:
    st.session_state.options = None
if "daily_log" not in st.session_state:
    st.session_state.daily_log = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚡ NutriLens AI")
    st.markdown("`system: active` — Deep Learning Vision & USDA Engine.")
    
    st.markdown("---")
    st.subheader("🎯 Daily Macro Goals")
    goal_calories = st.number_input("Target Calories (kcal)", value=2000, step=50)
    goal_protein = st.number_input("Target Protein (g)", value=130, step=5)
    
    total_logged_cals = sum(item.get("calories", 0) for item in st.session_state.daily_log)
    total_logged_protein = sum(item.get("protein", 0) for item in st.session_state.daily_log)
    
    st.markdown("---")
    st.markdown("### 📊 Today's Progress")
    st.progress(min(total_logged_cals / goal_calories, 1.0))
    st.text(f"Calories: {total_logged_cals} / {goal_calories} kcal")
    
    st.progress(min(total_logged_protein / goal_protein, 1.0))
    st.text(f"Protein: {total_logged_protein} / {goal_protein} g")
    
    if st.button("🗑️ Clear Daily Log", use_container_width=True):
        st.session_state.daily_log = []
        st.rerun()

# --- MAIN INTERFACE ---
st.title("Turning images into actionable nutrition insights.")
st.markdown("### 🍏 Zero-Shot Vision AI + USDA Database + Mistral LLM")

tab1, tab2, tab3 = st.tabs(["📸 Scan & Analyze", "📋 Daily Meal Log", "🧠 Model Transparency"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("1. Input Meal")
        uploaded = st.file_uploader("Upload food image (.jpg, .png)", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded Meal Preview", use_container_width=True)
            
        text_input = st.text_input("Or enter food name manually", placeholder="e.g., Grilled chicken breast with avocado")
        analyze_clicked = st.button("🚀 Analyze Meal", use_container_width=True)

    with col2:
        st.subheader("2. Results & Portion Adjustment")
        
        if (uploaded or text_input) and analyze_clicked:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)} if uploaded else None
            data = {"text": text_input} if text_input else None
            
            with st.spinner("🔍 Running zero-shot embeddings & querying database..."):
                try:
                    res = requests.post(f"{API_URL}/analyze-image", files=files, data=data, timeout=120)
                    res.raise_for_status()
                    resp_data = res.json()

                    if resp_data.get("status") == "need_confirmation":
                        st.session_state.state_id = resp_data["state_id"]
                        st.session_state.options = resp_data["options"]
                        st.session_state.result = None
                    elif resp_data.get("status") == "done":
                        st.session_state.result = resp_data
                        st.session_state.options = None
                    else:
                        st.error(resp_data.get("message", "Unable to analyze this food."))
                except Exception as exc:
                    st.error(f"Connection error: {exc}")

        # Disambiguation with Confidence Scores
        if st.session_state.options:
            st.warning("⚠️ Multiple food candidates detected. Select the correct match:")
            for opt in st.session_state.options:
                food_name = opt["food"]
                score = opt["score"]
                if st.button(f"👉 {food_name} (Similarity Score: {score:.2f})", use_container_width=True):
                    try:
                        with st.spinner("Fetching USDA data..."):
                            res = requests.post(
                                f"{API_URL}/confirm-food",
                                json={"state_id": st.session_state.state_id, "food": food_name},
                                timeout=120
                            )
                            res.raise_for_status()
                            data = res.json()
                            if data.get("status") == "done":
                                st.session_state.result = data
                                st.session_state.options = None
                                st.session_state.state_id = None
                                st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        # Final Result & Portion Scaler
        if st.session_state.result:
            r = st.session_state.result
            st.success(f"### 🍽️ Identified: {r['food']}")
            
            st.markdown("### ⚖️ Portion Size Multiplier")
            multiplier = st.slider("Adjust serving size multiplier", 0.5, 3.0, 1.0, 0.25)
            
            st.markdown("### 📈 Scaled Nutritional Breakdown")
            nutrition = r.get("nutrition", {})
            scaled_nutrition = {}
            
            if nutrition:
                cols = st.columns(min(len(nutrition), 3))
                idx = 0
                for k, v in nutrition.items():
                    try:
                        num_val = float(str(v).split()[0]) * multiplier
                        scaled_val = f"{num_val:.1f}"
                    except Exception:
                        scaled_val = v
                    scaled_nutrition[k] = scaled_val
                    
                    with cols[idx % len(cols)]:
                        st.metric(label=k, value=scaled_val)
                    idx += 1

            cals = 0
            prot = 0
            for k, v in scaled_nutrition.items():
                if "calorie" in k.lower():
                    try: cals = int(float(v))
                    except: pass
                if "protein" in k.lower():
                    try: prot = float(v)
                    except: pass

            if st.button("➕ Add to Daily Log", use_container_width=True):
                st.session_state.daily_log.append({
                    "food": f"{r['food']} ({multiplier}x)",
                    "calories": cals,
                    "protein": prot
                })
                st.success("Added to your daily log!")

            st.markdown("---")
            with st.expander("💡 AI Explanation & Recipe Recommendations", expanded=True):
                st.write(r["explanation"])

with tab2:
    st.subheader("📋 Today's Logged Meals")
    if not st.session_state.daily_log:
        st.info("No meals logged yet. Scan a meal in the 'Scan & Analyze' tab.")
    else:
        for idx, item in enumerate(st.session_state.daily_log):
            st.markdown(f"**{idx+1}. {item['food']}** — {item['calories']} kcal | {item['protein']}g Protein")

with tab3:
    st.subheader("🧠 Deep Learning Architecture & Transparency")
    st.markdown("""
    - **Zero-Shot Learning (CLIP):** Maps food images and text queries into a shared embedding space without explicit training per item.
    - **Vector Search (FAISS):** Performs fast approximate nearest neighbor search over thousands of food embeddings.
    - **Retrieval-Augmented Generation (Mistral LLM):** Combines structured USDA records with generative AI for explanations and recipes.
    """)
    if st.session_state.result:
        st.markdown(f"**Last Match Similarity Score:** `{st.session_state.result.get('score', 'N/A')}`")