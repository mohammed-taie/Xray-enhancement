import cv2
import numpy as np
import pywt
import streamlit as st
from io import BytesIO

# ============================
# 1. Streamlit UI Enhancements
# ============================
st.set_page_config(page_title="X-Ray Image Enhancement Tool", page_icon="🩺", layout="centered")

# Custom CSS for Background Color and Text Styling
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Open+Sans:wght@400;700&display=swap');

    /* Main content area */
    .stApp > div {
        background-color: #FFFFFF;
        font-family: 'Open Sans', sans-serif;
    }

    /* Sidebar (custom background color) */
    .css-1d391kg {
        background-color: #2E86C1;  /* Blue background for the sidebar */
        color: #FFFFFF;            /* White text for the sidebar */
        font-family: 'Roboto', sans-serif;
    }

    /* Sidebar header text */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #FFFFFF !important;
    }

    /* Main title styling */
    .main-title {
        font-family: 'Roboto', sans-serif;
        font-size: 48px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Subtitle styling */
    .subtitle {
        font-family: 'Open Sans', sans-serif;
        font-size: 24px;
        color: #1B4F72;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Divider styling */
    .divider {
        border: 2px solid #1B4F72;
        width: 60%;
        margin: auto;
        margin-bottom: 30px;
    }

    /* Uploader section styling */
    .uploader-section {
        text-align: center;
        font-size: 20px;
        color: #1B4F72;
        margin-bottom: 20px;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        font-size: 14px;
        color: #1B4F72;
        margin-top: 50px;
    }

    /* Dark Mode */
    .dark-mode {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .dark-mode .stApp > div {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .dark-mode .css-1d391kg {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .dark-mode .main-title {
        color: #FFFFFF;
    }
    .dark-mode .subtitle {
        color: #CCCCCC;
    }
    .dark-mode .uploader-section {
        color: #CCCCCC;
    }
    .dark-mode .footer {
        color: #CCCCCC;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Header with Logo
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Medical_Icon.svg/512px-Medical_Icon.svg.png", width=150)
st.sidebar.markdown("# **What Does This Tool Do?**")
st.sidebar.markdown("Enhance medical X-ray images using various techniques to improve contrast, sharpness, and visibility of critical details.")

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# Apply Dark Mode
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp > div {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .css-1d391kg {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .main-title {
            color: #FFFFFF;
        }
        .subtitle {
            color: #CCCCCC;
        }
        .uploader-section {
            color: #CCCCCC;
        }
        .footer {
            color: #CCCCCC;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main Title and Subtitle
st.markdown("""
    <div class="main-title">X-Ray Image Enhancement Tool 🩺</div>
    <div class="subtitle">Apply different image enhancement techniques for better visualization</div>
    <div class="divider"></div>
""", unsafe_allow_html=True)

# Uploader Section
st.markdown("""
    <div class="uploader-section">
        📂 <b>Upload an X-ray Image</b>
    </div>
""", unsafe_allow_html=True)

# ============================
# 2. Load Image with Error Handling
# ============================
def load_image(file_bytes):
    try:
        image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Invalid image file. Please upload a valid PNG, JPG, or JPEG.")
        if len(image.shape) != 2:
            raise ValueError("Please upload a grayscale image.")
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

# ============================
# 3. Image Enhancement Techniques
# ============================
def apply_histogram_equalization(gray_img):
    return cv2.equalizeHist(gray_img)

def apply_clahe(gray_img, clip_limit=3.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)

def apply_gamma_correction(gray_img, gamma=0.6):
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(gray_img, lookup_table)

def apply_bcet(gray_img):
    gray_img = gray_img.astype(np.float32)  # Prevent overflow
    Lmin, Lmax = np.min(gray_img), np.max(gray_img)
    Lmean, LMssum = np.mean(gray_img), np.mean(gray_img ** 2)
    Gmin, Gmax, Gmean = 0, 255, 110
    b = ((Lmax**2) * (Gmean - Gmin) - LMssum * (Gmax - Gmin) + (Lmin**2) * (Gmax - Gmean)) / \
        (2 * (Lmax * (Gmean - Gmin) - Lmean * (Gmax - Gmin) + Lmin * (Gmax - Gmean)))
    a = (Gmax - Gmin) / ((Lmax - Lmin) * (Lmax + Lmin - 2*b))
    c = Gmin - a * (Lmin - b) ** 2
    y = a * (gray_img - b) ** 2 + c
    return np.uint8(np.clip(y, 0, 255))

def apply_image_complement(gray_img):
    return cv2.bitwise_not(gray_img)

def apply_adaptive_gamma_correction(gray_img):
    mean_intensity = np.mean(gray_img) / 255.0
    gamma = np.log(0.5) / np.log(mean_intensity + 1e-6)  # Avoid log(0) error
    return apply_gamma_correction(gray_img, gamma=gamma)

def apply_laplacian_filter(gray_img):
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    return cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_unsharp_masking(gray_img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(gray_img, (0, 0), sigma)
    return cv2.addWeighted(gray_img, 1.0 + strength, blurred, -strength, 0)

def apply_wavelet_transform(gray_img, wavelet='haar', level=1, scale=0.5):
    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    coeffs = list(coeffs)
    coeffs[1:] = [(LH * scale, HL * scale, HH * scale) for LH, HL, HH in coeffs[1:]]
    return np.uint8(pywt.waverec2(coeffs, wavelet))

def apply_tophat_bothat(gray_img, kernel_size=(15, 15)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    tophat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    return cv2.addWeighted(tophat, 0.5, blackhat, 0.5, 0)

# ============================
# 4. File Uploader & Processing
# ============================
uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Technique Descriptions
technique_descriptions = {
    "Histogram Equalization": "Spreads out pixel intensities to improve global contrast.",
    "CLAHE": "Enhances local contrast while preventing over-amplification.",
    "Gamma Correction": "Adjusts brightness by non-linearly mapping pixel intensities.",
    "BCET": "Balances contrast dynamically based on image statistics.",
    "Image Complement": "Inverts pixel values, making dark areas bright and vice versa.",
    "Adaptive Gamma Correction": "Adjusts gamma dynamically based on image intensity distribution.",
    "Laplacian Filter": "Enhances edges by detecting intensity changes.",
    "Unsharp Masking": "Sharpens the image by amplifying edges and fine details.",
    "Wavelet Transform": "Enhances contrast and reduces noise while preserving details.",
    "Top-Hat Filtering": "Extracts bright regions to highlight small structures.",
    "Bottom-Hat Filtering": "Extracts dark regions to highlight small structures."
}

if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        image = load_image(file_bytes)
    except Exception as e:
        st.error(f"Error processing file '{uploaded_file.name}': {e}")
        st.stop()
    
    st.subheader(f"Processing: {uploaded_file.name}")
    
    # Side-by-Side Comparison
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="🖼️ Original Image", use_container_width=True)
    
    # Sidebar Configuration
    #st.sidebar.markdown("## ⚙️ **Settings**")
    
    # Enhancement Method Selection
    selected_method = st.sidebar.radio(
        "🎨 ****Select Enhancement Method****",
        [
            "Histogram Equalization",
            "CLAHE",
            "Gamma Correction",
            "BCET",
            "Image Complement",
            "Adaptive Gamma Correction",
            "Laplacian Filter",
            "Unsharp Masking",
            "Wavelet Transform",
            "Top-Hat Filtering",
            "Bottom-Hat Filtering"
        ],
        help="Choose an enhancement technique to improve image quality."
    )
    
    # Display Description of Selected Technique
    st.sidebar.markdown(f"#### **Description:**")
    st.sidebar.markdown(f"*{technique_descriptions[selected_method]}*")
    
    # Parameter Customization
    with st.sidebar.expander("🔧 **Adjust Parameters**"):
        if selected_method == "CLAHE":
            clip_limit = st.slider("Clip Limit", 1.0, 10.0, 3.0, help="Controls contrast enhancement intensity.")
            tile_grid_size = st.slider("Tile Grid Size", 2, 16, 8, help="Divides the image into smaller regions.")
        elif selected_method == "Gamma Correction":
            gamma = st.slider("Gamma Value", 0.1, 2.0, 0.6, help="Adjusts brightness and contrast.")
        elif selected_method == "Unsharp Masking":
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, help="Controls the blur radius.")
            strength = st.slider("Strength", 0.1, 3.0, 1.5, help="Controls the sharpening effect.")
        elif selected_method == "Wavelet Transform":
            wavelet = st.selectbox("Wavelet Type", ["haar", "db2", "sym4"], help="Select the wavelet type.")
            level = st.slider("Decomposition Level", 1, 5, 1, help="Number of decomposition levels.")
            scale = st.slider("Detail Coefficient Scale", 0.1, 2.0, 0.5, help="Scales detail coefficients for edge enhancement.")
    
    if st.sidebar.button("🚀 **Enhance Image**"):
        with st.spinner("Processing image..."):
            try:
                if selected_method == "CLAHE":
                    enhanced_img = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
                elif selected_method == "Gamma Correction":
                    enhanced_img = apply_gamma_correction(image, gamma=gamma)
                elif selected_method == "Unsharp Masking":
                    enhanced_img = apply_unsharp_masking(image, sigma=sigma, strength=strength)
                elif selected_method == "Wavelet Transform":
                    enhanced_img = apply_wavelet_transform(image, wavelet=wavelet, level=level, scale=scale)
                elif selected_method in ["Top-Hat Filtering", "Bottom-Hat Filtering"]:
                    enhanced_img = apply_tophat_bothat(image)
                else:
                    enhanced_img = globals()[f"apply_{selected_method.lower().replace(' ', '_')}"](image)
                
                with col2:
                    st.image(enhanced_img, caption=f"✨ {selected_method} Applied", use_container_width=True)
                
                # Success Message
                st.success("Image enhancement completed successfully!")
                
                # Download Button
                _, buffer = cv2.imencode(".png", enhanced_img)
                byte_io = BytesIO(buffer)
                st.download_button(
                    label="💾 **Download Enhanced Image**",
                    data=byte_io,
                    file_name=f"enhanced_{uploaded_file.name}",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error applying enhancement: {e}")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        🛠️ Built with Streamlit | 📧 Contact: support@xraytool.com
    </div>
""", unsafe_allow_html=True)