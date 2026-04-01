import streamlit as st
import requests
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# ---------------- SETUP ----------------
st.set_page_config(page_title="AgriAI System", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ---------------- LOAD MODEL ----------------
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="converted_keras/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open("converted_keras/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 AgriAI Navigation")

pages = ["Home", "Detection", "Forecast", "Advisory", "About"]

menu = st.sidebar.radio(
    "📌 Navigation",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = menu

# ---------------- HOME PAGE ----------------

if st.session_state.page == "Home":

    st.markdown("""
    <style>
    .card {
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 20px;
        background-color: #f9fff9;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        height: 100%;
    }
    .card-title {
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 10px;
    }
    .card-text {
        font-size: 15px;
        color: #333333;
    }
    .card:hover {
        transform: scale(1.03);
        transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""<h1 style='text-align:center; color:#2e7d32; font-weight:bold;'>🌿 Smart Agri AI System</h1>""", unsafe_allow_html=True)

    st.write("The Smart Agri AI System is an intelligent platform designed to support farmers in managing crop health effectively. It uses artificial intelligence to detect crop diseases instantly through image analysis, enabling early identification of problems. The system also analyzes weather data to predict future disease risks, helping farmers take preventive actions in advance. Additionally, it provides simple and practical farming advice in local languages like Hindi and Marathi, ensuring accessibility for all users. By combining detection, forecasting, and advisory in one platform, the system reduces dependency on guesswork and improves decision-making. It is designed to be fast, easy to use, and accessible through basic smartphones, making it suitable for real-world agricultural use.")

    # ✅ CARDS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔍 Detect Crop Disease</div>
            <div class="card-text">
            Upload or capture an image of the crop using your device, and the AI model will instantly analyze it to identify whether the plant is healthy or affected by disease. 
            This helps in early detection and reduces crop damage.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🌦 Predict Future Risk</div>
            <div class="card-text">
            The system analyzes weather data to predict possible disease outbreaks and classifies the risk level. 
            Farmers can take preventive measures in advance and protect their crops effectively.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧠 Expert Advisory</div>
            <div class="card-text">
            Get detailed farming guidance on crop management, fertilizers usage,irrigation and seasonl advice in local languages like Hindi and Marathi. 
            This helps farmers make better decisions easily.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⭐ Why Use AgriAI")
    st.write("✔ Easy to use\n\n ✔ Fast results\n\n ✔ Farmer friendly")

    lang = st.selectbox("🌐 Select Language", ["English", "Hindi", "Marathi"])

    if lang == "English":
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Go to the Detection tab from the sidebar menu  
2. Upload or capture crop image  
3. View instant AI result  
4. Check Forecast for risk  
5. Get Advisory in your language
        """)

    elif lang == "Hindi":
        st.markdown("### 📖 कैसे उपयोग करें")
        st.markdown("""
       1. साइडबार मेन्यू से Detection टैब पर जाएं  
2. फसल की तस्वीर अपलोड करें या कैमरा से फोटो लें  
3. तुरंत AI परिणाम देखें  
4. जोखिम जानने के लिए Forecast सेक्शन देखें  
5. अपनी भाषा में सलाह (Advisory) प्राप्त करें  
        """)

    elif lang == "Marathi":
        st.markdown("### 📖 कसे वापरावे")
        st.markdown("""
        1. साइडबार मेनूमधून Detection टॅब निवडा  
2. पिकाचा फोटो अपलोड करा किंवा कॅमेराने फोटो काढा  
3. लगेच AI परिणाम पहा  
4. धोका जाणून घेण्यासाठी Forecast विभाग पहा  
5. आपल्या भाषेत सल्ला (Advisory) मिळवा  
        """)
    
    st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    padding: 12px 30px;
    border-radius: 10px;
    border: none;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Get Started Now"):
            st.session_state.page = "Detection"
            st.rerun()

# ---------------- DETECTION PAGE ----------------

elif st.session_state.page == "Detection":

    st.markdown("<h2 style='text-align:center;'>🔍 Crop Disease Detection</h2>", unsafe_allow_html=True)

    option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

    image = None

    col1, col2 = st.columns([1,2])

    # -------- INPUT --------
    with col1:
        if option == "Upload Image":
            file = st.file_uploader("Upload Image", key="upload1")
            if file:
                image = Image.open(file).convert("RGB")

        else:
            file = st.camera_input("Take Photo", key="camera1")
            if file:
                image = Image.open(file).convert("RGB")

    # -------- SAFE BLOCK --------
    if image is not None:

        # Show image
        with col2:
            st.image(image, width=350)

        # -------- MODEL PROCESSING --------
        img = image.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_index = np.argmax(output_data)
        confidence = float(np.max(output_data)) * 100
        label = labels[class_index]

        # -------- LANGUAGE --------
        lang = st.selectbox("🌐 Select Result Language", ["English", "Hindi", "Marathi"])

        st.markdown("### 📊 Result")

        if lang == "English":
            st.success(f"Prediction: {label} ({confidence:.2f}%)")

        elif lang == "Hindi":
            st.success(f"परिणाम: {label} ({confidence:.2f}%)")

        elif lang == "Marathi":
            st.success(f"निकाल: {label} ({confidence:.2f}%)")

        # -------- SUGGESTIONS --------
        st.markdown("### 🌿 Suggestions")

        if "healthy" in label.lower():

            if lang == "English":
                st.success("""
✅ Crop is Healthy

• Continue regular monitoring of crops  
• Maintain proper irrigation schedule  
• Use balanced fertilizers  
• Keep field clean and weed-free  
• Follow seasonal farming practices  
""")

            elif lang == "Hindi":
                st.success("""
✅ फसल स्वस्थ है

• फसल की नियमित निगरानी करें  
• उचित सिंचाई बनाए रखें  
• संतुलित उर्वरकों का उपयोग करें  
• खेत को साफ और खरपतवार मुक्त रखें  
• मौसम के अनुसार खेती करें  
""")

            elif lang == "Marathi":
                st.success("""
✅ पीक निरोगी आहे

• पिकांची नियमित तपासणी करा  
• योग्य सिंचन करा  
• संतुलित खतांचा वापर करा  
• शेत स्वच्छ ठेवा व तण काढा  
• हंगामानुसार शेती करा  
""")

        else:

            if lang == "English":
                st.error("""
⚠ Crop is Diseased

• Remove infected leaves immediately  
• Avoid overwatering  
• Use recommended pesticides or fungicides  
• Maintain proper spacing between plants  
• Consult agricultural expert if condition worsens  
""")

            elif lang == "Hindi":
                st.error("""
⚠ फसल रोगग्रस्त है

• संक्रमित पत्तियों को तुरंत हटा दें  
• अधिक पानी देने से बचें  
• उचित कीटनाशकों या फफूंदनाशकों का उपयोग करें  
• पौधों के बीच उचित दूरी बनाए रखें  
• समस्या बढ़ने पर कृषि विशेषज्ञ से संपर्क करें  
""")

            elif lang == "Marathi":
                st.error("""
⚠ पीक आजारी आहे

• संक्रमित पाने लगेच काढून टाका  
• जास्त पाणी देऊ नका  
• योग्य कीटकनाशक किंवा बुरशीनाशक वापरा  
• पिकांमध्ये योग्य अंतर ठेवा  
• समस्या वाढल्यास तज्ञांचा सल्ला घ्या  
""")

    else:
        st.info("📷 Please upload or capture an image to continue")
# ---------------- FORECAST PAGE ----------------
elif st.session_state.page == "Forecast":

    st.markdown("<h2 style='text-align:center; color:#2e7d32;'>🌦 Weather Forecast</h2>", unsafe_allow_html=True)

    url = "https://api.open-meteo.com/v1/forecast?latitude=18.52&longitude=73.85&daily=temperature_2m_max,temperature_2m_min&timezone=auto"

    data = requests.get(url).json()

    days = data["daily"]["time"]
    max_temp = data["daily"]["temperature_2m_max"]
    min_temp = data["daily"]["temperature_2m_min"]

    for i in range(5):

        avg = (max_temp[i] + min_temp[i]) / 2

        if avg > 30:
            risk = "🔴 High Risk"
        elif avg > 25:
            risk = "🟡 Medium Risk"
        else:
            risk = "🟢 Low Risk"

        st.markdown(f"""
        <div style='border:1px solid green;padding:15px;border-radius:10px;margin-bottom:10px'>
        📅 {days[i]} <br>
        🌡 Max: {max_temp[i]}°C <br>
        ❄ Min: {min_temp[i]}°C <br>
        ⚠ Risk: {risk}
        </div>
        """, unsafe_allow_html=True)

# ---------------- ADVISORY PAGE ----------------
elif st.session_state.page == "Advisory":

    st.markdown("<h2 style='text-align:center; color:#2e7d32;'>🧠 Smart Advisory</h2>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .topic-box {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9fff9;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .topic-title {
        font-size: 18px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    lang = st.selectbox("🌐 Select Language", ["English", "Hindi", "Marathi"])

    # ---------------- ENGLISH ----------------
    if lang == "English":

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌾 Crop Management</div>
            Maintain proper spacing between crops to ensure healthy growth. Remove infected plants early and monitor crops regularly.
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌱 Fertilizer Usage</div>
            Use balanced fertilizers based on soil testing. Avoid excessive nitrogen usage.
            </div>""", unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌍 Soil Health</div>
            Maintain soil fertility using organic matter and avoid excessive chemicals.
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">💧 Irrigation Tips</div>
            Avoid overwatering and use drip irrigation. Water crops in early morning.
            </div>""", unsafe_allow_html=True)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌿 Seasonal Advice</div>
            Follow seasonal crop patterns and decide farming based on humidity and temperature.
            </div>""", unsafe_allow_html=True)

        with col6:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🛡 Pest Control</div>
            Use eco-friendly pest control methods like neem-based sprays.
            Use Organic manure and fertilizer for farming.
            </div>""", unsafe_allow_html=True)

    # ---------------- HINDI ----------------
    elif lang == "Hindi":

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌾 फसल प्रबंधन</div>
            फसलों के बीच उचित दूरी रखें और संक्रमित पौधों को समय पर हटाएं।
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌱 उर्वरक उपयोग</div>
            मिट्टी परीक्षण के आधार पर संतुलित उर्वरकों का उपयोग करें।
            </div>""", unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌍 मिट्टी स्वास्थ्य</div>
            मिट्टी में जैविक पदार्थ बनाए रखें और रसायनों का अधिक उपयोग न करें।
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">💧 सिंचाई सुझाव</div>
            अधिक पानी से बचें और सुबह सिंचाई करें।
            </div>""", unsafe_allow_html=True)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌿 मौसमी सलाह</div>
            आर्द्रता और तापमान के आधार पर खेती की योजना बनाएं।
            </div>""", unsafe_allow_html=True)

        with col6:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🛡 कीट नियंत्रण</div>
            जैविक तरीकों का उपयोग करें और कीटनाशक कम करें।
            </div>""", unsafe_allow_html=True)

    # ---------------- MARATHI ----------------
    elif lang == "Marathi":

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌾 पीक व्यवस्थापन</div>
            पिकांमध्ये योग्य अंतर ठेवा आणि संक्रमित झाडे काढून टाका.
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌱 खतांचा वापर</div>
            माती तपासणीवर आधारित खतांचा वापर करा.
            </div>""", unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌍 माती आरोग्य</div>
            सेंद्रिय पदार्थ वापरून माती सुपीक ठेवा.
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">💧 सिंचन टिप्स</div>
            जास्त पाणी देऊ नका आणि सकाळी पाणी द्या.
            </div>""", unsafe_allow_html=True)

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🌿 हंगामी सल्ला</div>
            आर्द्रता आणि तापमानानुसार शेतीचे नियोजन करा.
            </div>""", unsafe_allow_html=True)

        with col6:
            st.markdown("""<div class="topic-box">
            <div class="topic-title">🛡 कीड नियंत्रण</div>
            सेंद्रिय पद्धती वापरा आणि रासायनिक फवारणी कमी करा.
            </div>""", unsafe_allow_html=True)
# ---------------- ABOUT PAGE ----------------
elif st.session_state.page == "About":

    # ✅ CSS for boxes
    st.markdown("""
    <style>
    .about-box {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        background-color: #f9fff9;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .about-title {
        font-size: 18px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ✅ GREEN HEADING
    st.markdown("""
    <h2 style='text-align:center; color:#2e7d32;'>ℹ About AgriAI System</h2>
    """, unsafe_allow_html=True)

    # ✅ DESCRIPTION BOX
    st.markdown("""
    <div class="about-box">
    AgriAI System is an innovative platform designed to empower farmers by integrating artificial intelligence into agriculture. 
    It helps farmers detect crop diseases at an early stage using image-based AI models, reducing crop loss and improving productivity. 
    The system also provides weather-based forecasting to predict potential disease risks, allowing farmers to take preventive actions in advance. 
    In addition, it offers expert farming advisory in multiple local languages such as Hindi and Marathi, ensuring accessibility for all users. 
    By combining detection, forecasting, and advisory in one platform, AgriAI simplifies decision-making and supports sustainable farming practices. 
    The platform is designed to be simple, fast, and usable even on basic smartphones, making it practical for real-world agricultural use.
    </div>
    """, unsafe_allow_html=True)

    # ✅ KEY FEATURES
    st.markdown("<h3 style='color:#2e7d32;'>🌟 Key Features</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
    <div class="about-title">🤖 AI-Powered Detection</div>
    The system uses advanced AI models to analyze crop images and instantly detect whether the plant is healthy or diseased. 
    This helps in early identification and reduces crop damage significantly.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
    <div class="about-title">🌐 Multilingual Support</div>
    AgriAI provides advisory in multiple languages like English, Hindi, and Marathi, ensuring that farmers from different regions can easily understand and use the system.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
    <div class="about-title">🧠 Expert Advisory</div>
    The platform offers detailed guidance on crop management, fertilizer usage, irrigation, and pest control, helping farmers make better and informed decisions.
    </div>
    """, unsafe_allow_html=True)

    # ✅ OUR VALUES
    st.markdown("<h3 style='color:#2e7d32;'>🌱 Our Values</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="about-box">
        <div class="about-title">🌍 Sustainability</div>
        We promote eco-friendly farming practices that reduce chemical usage and protect soil health. 
        Our goal is to ensure long-term agricultural productivity while maintaining environmental balance and supporting future generations of farmers.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-box">
        <div class="about-title">🌐 Accessibility</div>
        We believe that technology should be accessible to every farmer, regardless of location or technical knowledge. 
        AgriAI is designed to be simple, user-friendly, and usable on basic devices with support for local languages.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="about-box">
        <div class="about-title">🚀 Innovation</div>
        We continuously aim to bring modern AI-driven solutions into agriculture to solve real-world farming problems. 
        Our focus is on creating smart, efficient, and scalable tools that improve farming practices and productivity.
        </div>
        """, unsafe_allow_html=True)