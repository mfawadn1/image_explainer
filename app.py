import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import requests
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Image Captioner & Explainer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #1e40af;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .caption-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .info-box {
        background: #e0f2fe;
        padding: 15px;
        border-left: 4px solid #0284c7;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching to avoid reloading
@st.cache_resource
def load_model():
    """Load the BLIP model and processor once and cache it"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(device)
    return processor, model, device

# Initialize model
processor, model, device = load_model()

def resize_image(image, max_size=800):
    """Resize image to max dimension while maintaining aspect ratio"""
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def generate_caption(image, detail_level="Short"):
    """Generate caption and explanation for the image"""
    # Resize image for processing
    processed_image = resize_image(image)
    
    # Generate short caption
    inputs = processor(processed_image, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = model.generate(**inputs, max_length=50)
    short_caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    
    # Generate detailed explanation based on detail level
    if detail_level == "Short":
        prompt = "a detailed description:"
        max_length = 75
    elif detail_level == "Medium":
        prompt = "a comprehensive description including colors, actions, and context:"
        max_length = 100
    else:  # Detailed
        prompt = "an in-depth analysis of this image, including all visible elements, colors, actions, setting, context, and possible interpretations:"
        max_length = 150
    
    inputs = processor(processed_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        explanation_ids = model.generate(**inputs, max_length=max_length)
    detailed_explanation = processor.decode(explanation_ids[0], skip_special_tokens=True)
    
    return short_caption, detailed_explanation

def add_caption_to_image(image, caption):
    """Add caption text overlay to image with better readability"""
    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    
    # Calculate font size based on image dimensions (larger font)
    img_width, img_height = img_copy.size
    font_size = max(30, min(60, img_width // 15))  # Increased font size
    
    # Try to use a better font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Split caption into multiple lines if too long
    words = caption.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] < img_width - 40:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate total text height
    total_height = 0
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        total_height += line_height + 5
    
    # Calculate starting position (bottom center)
    y_start = img_height - total_height - 30
    
    # Draw background for all lines
    padding = 15
    background_coords = [
        0,
        y_start - padding,
        img_width,
        img_height
    ]
    draw.rectangle(background_coords, fill=(0, 0, 0, 200))
    
    # Draw each line
    current_y = y_start
    for line, line_height in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (img_width - text_width) // 2
        draw.text((x, current_y), line, font=font, fill=(255, 255, 255, 255))
        current_y += line_height + 5
    
    return img_copy

def fetch_image_from_url(url):
    """Fetch image from URL with better error handling"""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the image
        response = requests.get(url, timeout=10, headers=headers, stream=True)
        response.raise_for_status()
        
        # Check if content type is an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            raise Exception(f"URL does not point to an image. Content-Type: {content_type}")
        
        # Open image
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except requests.exceptions.MissingSchema:
        raise Exception("Invalid URL format. Please include http:// or https://")
    except requests.exceptions.ConnectionError:
        raise Exception("Could not connect to the URL. Please check your internet connection.")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The server took too long to respond.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"HTTP Error: {e.response.status_code}. The image could not be fetched.")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def get_image_download_link(img, filename, text):
    """Generate download link for image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" style="text-decoration:none;"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; margin: 5px;">📥 {text}</button></a>'
    return href

# Sidebar
st.sidebar.title("📝 Instructions")
st.sidebar.markdown("""
<div style='background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
<b>How to use this app:</b><br><br>
1️⃣ <b>Select input method</b><br>
2️⃣ <b>Provide an image</b>:<br>
   • Upload a file<br>
   • Use camera<br>
   • Enter image URL<br>
3️⃣ <b>Adjust detail level</b><br>
4️⃣ <b>Download results</b><br><br>
<i>💡 First run may take time to load the model.</i>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Input method selector
input_method = st.sidebar.selectbox(
    "🎯 Select Input Method",
    ["Upload Image", "Camera Capture", "Image URL"]
)

# Detail level selector
detail_level = st.sidebar.selectbox(
    "🔍 Detail Level",
    ["Short", "Medium", "Detailed"],
    index=1
)

# Main app
st.markdown("<h1>🖼️ Image Captioner & Explainer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #64748b;'>Generate AI-powered captions and detailed explanations for your images</p>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

image = None

# Handle different input methods
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

elif input_method == "Camera Capture":
    camera_image = st.camera_input("📷 Take a picture")
    if camera_image is not None:
        try:
            image = Image.open(camera_image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            st.error(f"❌ Error loading camera image: {str(e)}")

else:  # Image URL
    st.markdown("<div class='info-box'>💡 <b>Tip:</b> Right-click on any image online and select 'Copy Image Address' to get the URL</div>", unsafe_allow_html=True)
    url = st.text_input("🔗 Enter image URL", placeholder="https://example.com/image.jpg")
    
    if url:
        if url.strip():  # Check if URL is not empty
            try:
                with st.spinner("🌐 Fetching image from URL..."):
                    image = fetch_image_from_url(url.strip())
                    st.success("✅ Image loaded successfully!")
            except Exception as e:
                st.error(f"❌ {str(e)}")
                st.info("🔍 **Troubleshooting:**\n- Make sure the URL ends with .jpg, .jpeg, or .png\n- Check if the URL is accessible\n- Try a different image URL")

# Process image if available
if image is not None:
    # Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>📸 Original Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    # Generate captions
    with st.spinner("🤖 AI is analyzing your image... Please wait..."):
        try:
            short_caption, detailed_explanation = generate_caption(image, detail_level)
            captioned_image = add_caption_to_image(image, short_caption)
            
            with col2:
                st.markdown("<h3>🏷️ Image with Caption</h3>", unsafe_allow_html=True)
                st.image(captioned_image, use_container_width=True)
            
            # Download buttons
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>💾 Download Options</h3>", unsafe_allow_html=True)
            
            col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
            
            with col_d1:
                st.markdown(get_image_download_link(image, "original_image.png", "Download Original"), unsafe_allow_html=True)
            
            with col_d2:
                st.markdown(get_image_download_link(captioned_image, "captioned_image.png", "Download with Caption"), unsafe_allow_html=True)
            
            with col_d3:
                # Create a text file with caption and explanation
                text_content = f"SHORT CAPTION:\n{short_caption}\n\nDETAILED EXPLANATION:\n{detailed_explanation}"
                b64 = base64.b64encode(text_content.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="image_description.txt" style="text-decoration:none;"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; margin: 5px;">📥 Download Text</button></a>'
                st.markdown(href, unsafe_allow_html=True)
            
            # Display results
            st.markdown("---")
            
            st.markdown("<h3>📝 Short Caption</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='caption-box' style='font-size: 20px; font-weight: 500; color: #1e40af;'>{short_caption}</div>", unsafe_allow_html=True)
            
            st.markdown("<h3>📖 Detailed Explanation</h3>", unsafe_allow_html=True)
            with st.expander("Click to view detailed explanation", expanded=True):
                st.markdown(f"<div style='font-size: 18px; line-height: 1.6; color: #475569;'>{detailed_explanation}</div>", unsafe_allow_html=True)
            
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

else:
    # Placeholder message
    st.markdown("""
        <div style='background: white; padding: 40px; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0;'>
            <h2 style='color: #1e40af;'>👆 Get Started</h2>
            <p style='font-size: 18px; color: #64748b;'>Please provide an image using one of the methods in the sidebar</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show example
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>💡 Example Use Cases</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-size: 40px;'>📚</div>
                <h4>Photo Organization</h4>
                <p style='font-size: 14px; color: #64748b;'>Auto-generate descriptions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-size: 40px;'>♿</div>
                <h4>Accessibility</h4>
                <p style='font-size: 14px; color: #64748b;'>Create alt text</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-size: 40px;'>✍️</div>
                <h4>Content Creation</h4>
                <p style='font-size: 14px; color: #64748b;'>Get descriptions</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <div style='font-size: 40px;'>🎓</div>
                <h4>Education</h4>
                <p style='font-size: 14px; color: #64748b;'>Analyze images</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <p style='color: #64748b; margin: 0;'>⚡ Powered by <b>Salesforce BLIP Model</b> | Built with <b>Streamlit</b></p>
        <p style='color: #94a3b8; font-size: 14px; margin: 5px 0 0 0;'>AI-powered image understanding at your fingertips</p>
    </div>
""", unsafe_allow_html=True)
