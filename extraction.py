import ocr
from dotenv import load_dotenv
import os
import cv2
from PIL import Image
import json

load_dotenv(dotenv_path=".env")  # take environment variables from .env.

receipt_extractor = ocr.ReceiptOCRDonut("naver-clova-ix/donut-base-finetuned-cord-v2")
invoice_extractor = ocr.InvoiceOCRDonut("katanaml-org/invoices-donut-model-v1")
gemini_extractor = ocr.GoogleGeminiExtractor(os.getenv("GOOGLE_API_KEY"))

# tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Path to the tesseract executable (if needed)
# ocr = TesseractOCR()
tesseract_ocr = ocr.TesseractOCR()
hdr_model = ocr.HDR('./models/resnet164.h5')
easy_ocr = ocr.EasyOCR(languages=['id'])
text_extractor = ocr.TextExtractor(hdr_model, tesseract_ocr, easy_ocr)


def extract_receipt(image):
    preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_img = Image.fromarray(preprocessed_img)
    result = receipt_extractor.predict(preprocessed_img)
    json_string = json.dumps(result, indent=4)
    return json_string


def extract_invoice(image):
    preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_img = Image.fromarray(preprocessed_img)
    result = invoice_extractor.predict(preprocessed_img)
    json_string = json.dumps(result, indent=4)
    return json_string


def extract_with_gemini(image, fields):
    preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_img = Image.fromarray(preprocessed_img)
    result = gemini_extractor.extract_information(preprocessed_img, fields)
    json_string = json.dumps(result, indent=4)
    return json_string


def extract_with_bounding_boxes(image, ocr_locations):
    result = text_extractor.extract_text(image, ocr_locations)
    json_string = json.dumps(result, indent=4)
    return json_string
