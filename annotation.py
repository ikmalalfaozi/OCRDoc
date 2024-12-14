import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image

# st.set_page_config(layout="wide")

def get_bounding_boxes(image: np.ndarray, key):
    st.title("Bounding Box Annotation Tool")

    # Convert the image to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to fit in the canvas
    max_width = 1200
    max_height = 1200
    h, w, _ = img.shape

    scale_ratio = min(max_width / w, max_height / h)
    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Display the image using st_canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=Image.fromarray(resized_img),
        update_streamlit=True,
        height=new_h,
        width=new_w,
        drawing_mode="rect",
        key="canvas",
    )

    st.divider()

    # Process the result
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        bounding_boxes = []
        st.write("Bounding Boxes:")
        for i, obj in enumerate(objects):
            if obj["type"] == "rect":
                left = int(obj["left"] / scale_ratio)
                top = int(obj["top"] / scale_ratio)
                width = int(obj["width"] / scale_ratio)
                height = int(obj["height"] / scale_ratio)
                bbox = (left, top, width, height)
                bounding_boxes.append(bbox)

            # Create a form for user to input id, filter_keywords, and model
            with st.form(key=f'bbox_form_{i}'):
                st.write(f"Bounding Box {i + 1}")
                st.text_input(f"ID for Bounding Box {i + 1} (required)", key=f'id_{i}', value=f'id_{i}')
                st.text_input(f"Filter Keywords for Bounding Box {i + 1} (comma-separated)",
                                                key=f'keywords_{i}')
                st.selectbox(f"Model for Bounding Box {i + 1}", ["hdr", "tesseract", "easyocr"],
                                     key=f'model_{i}')
                st.form_submit_button(label="Submit")

        # Save bounding boxes information
        if st.button("Save Bounding Boxes Info"):
            bbox_infos = []
            for i, bbox in enumerate(bounding_boxes):
                bbox_id = st.session_state.get(f'id_{i}', '')
                filter_keywords = st.session_state.get(f'keywords_{i}', '')
                model = st.session_state.get(f'model_{i}', '')
                bbox_info = {
                    "id": bbox_id,
                    "bbox": bbox,
                    "filter_keywords": [kw.strip() for kw in filter_keywords.split(',')],
                    "model": model
                }
                bbox_infos.append(bbox_info)
            return bbox_infos

# Example usage:
if __name__ == "__main__":
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Call the function
        bbox_infos = get_bounding_boxes(img, os.path.splitext(uploaded_file.name)[0])
        if bbox_infos:
            st.json(bbox_infos)
