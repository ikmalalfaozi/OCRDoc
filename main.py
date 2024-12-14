import streamlit as st
import os
import numpy as np
import cv2

import ocr
import scan
import extraction
import annotation

st.set_page_config(layout="wide")

def main():
    st.title("OCRDoc")

    # Choose action: scan or extract text
    action = st.selectbox("Choose action:", ["Scan Image", "Extract Text", "Extract Information"])

    if action == "Scan Image":
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # Scan the image using the selected scanner
            scan_option = st.selectbox("Choose scanner type", ["Line Scanner", "Contour Scanner", "Segmentation Scanner"])
            if st.button("Scan"):
                scanned_img = scan.scan(img, scan_option)

                # Display scanned image
                st.image(cv2.cvtColor(scanned_img, cv2.COLOR_BGR2RGB), caption='Scanned Image.', use_column_width=True)

                # Convert the scanned image to bytes
                is_success, buffer = cv2.imencode(".jpg", scanned_img)
                if is_success:
                    scanned_img_bytes = buffer.tobytes()

                    # Add a download button for the scanned image
                    st.download_button(
                        label="Download Scanned Image",
                        data=scanned_img_bytes,
                        file_name="scanned_image.jpg",
                        mime="image/jpeg"
                    )

    elif action == "Extract Text":
        # upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # Scan before text extraction (optional)
            scan_before_extraction = st.checkbox("Scan image before text extraction?")
            if scan_before_extraction:
                # Scan the image using the selected scanner
                scan_option = st.selectbox("Choose scanner type",
                                           ["Line Scanner", "Contour Scanner", "Segmentation Scanner"])

            if scan_before_extraction:
                img_to_extract = scan.scan(img, scan_option)
            else:
                img_to_extract = img

            bbox_infos = annotation.get_bounding_boxes(img_to_extract, os.path.splitext(uploaded_file.name)[0])
            if bbox_infos:
                st.json(bbox_infos)
                ocr_locations = []
                for bbox_info in bbox_infos:
                    ocr_location = ocr.OCRLocation(
                        id=bbox_info["id"],
                        bbox=bbox_info["bbox"],
                        filter_keywords=bbox_info["filter_keywords"],
                        model=bbox_info["model"]
                    )
                    ocr_locations.append(ocr_location)

                result = extraction.extract_with_bounding_boxes(img_to_extract, ocr_locations)
                st.json(result)


    elif action == "Extract Information":
        # upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)

            # Scan before text extraction (optional)
            scan_before_extraction = st.checkbox("Scan image before information extraction?")
            if scan_before_extraction:
                # Scan the image using the selected scanner
                scan_option = st.selectbox("Choose scanner type",
                                           ["Line Scanner", "Contour Scanner", "Segmentation Scanner"])

            # Model option
            model_option = st.selectbox("Choose Model",
                                        ["Receipt Donut", "Invoice Donut", "Gemini Pro Vision"])

            # defines the fields to be extracted
            if model_option == "Gemini Pro Vision":
                st.header("Define Fields")

                fields = {}
                for i in range(st.number_input("Number of fields:", min_value=1)):
                    field_name = st.text_input(f"Field Name ({i + 1}):")
                    fields[field_name] = ""

            # Extract information
            if st.button("Extract Information"):
                if scan_before_extraction:
                    img_to_extract = scan.scan(img, scan_option)
                else:
                    img_to_extract = img

                # extract text and information with the selected model
                if model_option == "Receipt Donut":
                    result = extraction.extract_receipt(img_to_extract)
                    st.json(result)
                elif model_option == "Invoice Donut":
                    result = extraction.extract_invoice(img_to_extract)
                    st.json(result)
                elif model_option == "Gemini Pro Vision":
                    result = extraction.extract_with_gemini(img_to_extract, fields)
                    st.json(result)


if __name__ == "__main__":
    main()
