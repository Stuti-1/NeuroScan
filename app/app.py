import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: white;
    }
    
    .prediction-result {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Upload area styling */
    .upload-container {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_tumor_model():
    try:
        model = load_model('../models/tumor_detector.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize model
model = load_tumor_model()

# Class names and their descriptions
class_info = {
    'Glioma': {
        'description': 'A type of brain tumor that begins in glial cells',
        'color': '#ff6b6b',
        'icon': '🔴'
    },
    'Meningioma': {
        'description': 'A tumor that arises from the meninges',
        'color': '#4ecdc4',
        'icon': '🟢'
    },
    'No Tumor': {
        'description': 'No tumor detected in the MRI scan',
        'color': '#45b7d1',
        'icon': '✅'
    },
    'Pituitary': {
        'description': 'A tumor in the pituitary gland',
        'color': '#f9ca24',
        'icon': '🟡'
    }
}

class_names = list(class_info.keys())

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 Brain Tumor Detection</h1>
    <p class="header-subtitle">Advanced AI-powered MRI analysis for medical diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 About This App")
    st.markdown("""
    This application uses a deep learning model to analyze MRI brain scans and detect potential tumors.
    
    **Supported Tumor Types:**
    - 🔴 **Glioma** - Glial cell tumors
    - 🟢 **Meningioma** - Meningeal tumors  
    - 🟡 **Pituitary** - Pituitary gland tumors
    - ✅ **No Tumor** - Healthy brain tissue
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Model Information")
    st.info("Deep Learning CNN Model\nAccuracy: ~95%\nTraining Images: 7,000+")
    
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning("This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.")

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose an MRI image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        # Image information
        width, height = img.size
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Width", f"{width}px")
        with col_info2:
            st.metric("Height", f"{height}px")
        with col_info3:
            st.metric("Size", f"{file_size:.1f}KB")

with col2:
    if uploaded_file is not None and model is not None:
        st.markdown("### 🔍 Analysis Results")
        
        # Add a prediction button
        if st.button("🚀 Analyze MRI Scan", use_container_width=True):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            status_text.text("🔄 Preprocessing image...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            # Preprocess image
            img_resized = img.resize((150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            status_text.text("🧠 Running AI analysis...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = class_names[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx] * 100
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            class_color = class_info[predicted_class]['color']
            class_icon = class_info[predicted_class]['icon']
            class_desc = class_info[predicted_class]['description']
            
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 3rem;">{class_icon}</div>
                <div class="prediction-result">{predicted_class}</div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.8;">
                    {class_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence breakdown chart
            st.markdown("#### 📊 Confidence Breakdown")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=prediction[0] * 100,
                    marker_color=[class_info[name]['color'] for name in class_names],
                    text=[f"{val:.1f}%" for val in prediction[0] * 100],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Prediction Confidence for Each Class",
                xaxis_title="Tumor Type",
                yaxis_title="Confidence (%)",
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            fig.update_traces(
                textfont_size=12,
                textfont_color="white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("#### 💡 Insights")
            
            if confidence > 90:
                st.success(f"🎯 High confidence prediction ({confidence:.1f}%)")
            elif confidence > 70:
                st.warning(f"⚠️ Moderate confidence prediction ({confidence:.1f}%)")
            else:
                st.error(f"🔍 Low confidence prediction ({confidence:.1f}%) - Consider additional analysis")
            
            # Top 2 predictions
            top_2_indices = np.argsort(prediction[0])[::-1][:2]
            st.markdown("**Top 2 Predictions:**")
            for i, idx in enumerate(top_2_indices, 1):
                class_name = class_names[idx]
                conf = prediction[0][idx] * 100
                icon = class_info[class_name]['icon']
                st.write(f"{i}. {icon} **{class_name}**: {conf:.1f}%")
    
    elif model is None:
        st.error("🚫 Model failed to load. Please check the model path.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏥 <strong>Brain Tumor Detection System</strong> | Powered by Deep Learning</p>
    <p><em>For educational and research purposes only</em></p>
</div>
""", unsafe_allow_html=True)