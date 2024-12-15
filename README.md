# OCRDoc: Optical Character Recognition for Document Processing

## Overview
OCRDoc is a comprehensive Optical Character Recognition (OCR) solution designed for document text extraction and preprocessing. The system integrates various OCR models and preprocessing techniques to enhance text extraction accuracy from diverse document types.

---

## Features
- **Modular Design**: Includes distinct modules for scanning and OCR.
- **Preprocessing Tools**: Correction for rotation, distortion, and alignment.
- **Flexible OCR Options**: Supports structured, semi-structured, and unstructured document types.
- **User Interface**: Streamlit-based web interface for ease of use.  
  **Note**: The UI is designed for convenience but has limited features compared to direct Python module usage.

---

## Modules

### **1. Scanner Module**
Responsible for document preprocessing to correct rotation, alignment, and distortion.

#### Submodules:
1. **Line Scanner**
   - Detects and aligns documents based on line detection algorithms.

2. **Contour Scanner**
   - Identifies document edges and adjusts the perspective.

3. **SIFT Feature Matcher**
   - Utilizes Scale-Invariant Feature Transform (SIFT) to detect and align documents.

4. **ORB Feature Matcher**
   - Employs Oriented FAST and Rotated BRIEF (ORB) for feature-based alignment.

5. **Document Segmentation Scanner**
   - Pretrained Models for Document Segmentation:
     - **DeepLabv3** (Backbone: MobileNetV3-Large)  
       **Model Link**: [Download DeepLabv3 Model](https://drive.google.com/file/d/1pMS8N3JR-o0cLRohPm77L8eRqMo9WntX/view?usp=sharing)

**Example Usage**: For detailed Python examples of how to use the Scanner module, see the [Scanner README](scanner_usage.md).

---

### **2. OCR Module**
Handles text extraction from processed document images using multiple OCR models.

#### Submodules:
1. **Invoice OCR (Donut)**
   - Extracts structured information from invoices.

2. **Receipt OCR (Donut)**
   - Extracts structured information from receipts.

3. **Google Gemini Extractor**
   - Uses Google Gemini AI for extracting specific information from documents.

4. **EasyOCR**
   - General-purpose OCR for extracting text from images.

5. **Tesseract OCR**
   - Open-source OCR for simple text recognition.

6. **HDR (Handwritten Digit Recognition)**
   - Custom model for recognizing handwritten digits.
     - [MobileNet](https://drive.google.com/file/d/1BUK1wcKOVjrR4-abYT1vZ89YB_3tWtLU/view?usp=sharing) 
     - [ResNet164](https://drive.google.com/file/d/1IVuuCKhvjqIUzoKNZU7nT5_1cgxiKZfv/view?usp=sharing) 
     - [VGG16](https://drive.google.com/file/d/1hRKC656sVu_YPW-Mg6iJH2VcECsPD0wG/view?usp=sharing)   

7. **Text Extractor**
   - Combines multiple OCR approaches (Tesseract, EasyOCR, HDR) for bounding-box-based text extraction.

**Example Usage**: For detailed Python examples of how to use the OCR module, see the [OCR README](ocr_usage.md).

---

## User Interface
Implemented with **Streamlit**, the UI provides:
- **File Upload**: Users can upload document images.
- **Preprocessing Options**: Select methods for correcting rotation, distortion, or alignment.
- **OCR Model Selection**: Choose from Donut, Google Gemini, Text Extractor (Combine EasyOCR, Tesseract, and HDR).
- **Output Visualization**: Display extracted text in JSON format.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ikmalalfaozi/OCRDoc.git
   cd OCRDoc
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download models:
   - Document Segmentation Model
     - [DeepLabv3 Model](https://drive.google.com/file/d/1pMS8N3JR-o0cLRohPm77L8eRqMo9WntX/view?usp=sharing)
   - HDR Model:
     - [MobileNet](https://drive.google.com/file/d/1BUK1wcKOVjrR4-abYT1vZ89YB_3tWtLU/view?usp=sharing) 
     - [ResNet164](https://drive.google.com/file/d/1IVuuCKhvjqIUzoKNZU7nT5_1cgxiKZfv/view?usp=sharing) 
     - [VGG16](https://drive.google.com/file/d/1hRKC656sVu_YPW-Mg6iJH2VcECsPD0wG/view?usp=sharing)

4. Place the models in the appropriate directory (e.g., `models/`).

---

## Usage
1. Launch the Streamlit interface:
   ```bash
   streamlit run app.py
   ```

2. Upload a document image.

3. Select preprocessing options (e.g., Line Scanner, Contour Scanner).

4. Choose the OCR model for text extraction (e.g., Donut, Gemini, HDR).

5. View extracted text and information in JSON format.

---

## Example Usage

Detailed usage examples for each module are available in the following files:
- **Scanner Module**: [Scanner README](./scanner/README.md)
- **OCR Module**: [OCR README](./ocr/README.md)

---

## Contributions
We welcome contributions! Please submit issues or pull requests to enhance OCRDoc.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For further inquiries, please contact [ikmalalfaozi@gmail.com](mailto:ikmalalfaozi@gmail.com).

---
