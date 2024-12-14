from collections import namedtuple
from typing import List
import cv2

OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords", "model"])

class TextExtractor:
    def __init__(self, hdr_model, tesseract_ocr, easy_ocr):
        """
        Initialize the TextExtractor class.

        Args:
            hdr_model: HDR object, instance of the HDR class for digit recognition
            tesseract_ocr: TesseractOCR object, instance of the TesseractOCR class for OCR using Tesseract
            easy_ocr: EasyOCR object, instance of the EasyOCR class for OCR using EasyOCR
        """
        self.hdr_model = hdr_model
        self.tesseract_ocr = tesseract_ocr
        self.easy_ocr = easy_ocr

    def extract_text(self, image, ocr_locations):
        """
        Extract text from specified locations in an image.

        Args:
            image: str or 2-D array, path to the image file or input image
            ocr_locations: list of OCRLocation, locations to extract text with specified configurations

        Returns:
            dict: extracted texts with location IDs as keys
        """
        if isinstance(image, str):
            # Load the image if path is provided
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # Use the image directly if it's already in numpy array format
            img = image

        # Initialize results dictionary
        extracted_texts = {}

        # Process each OCR location
        for loc in ocr_locations:
            # extract the OCR ROI from the image
            (x, y, w, h) = loc.bbox
            roi = img[y:y + h, x:x + w]
            if loc.model == 'hdr':
                extractor = self.hdr_model
            elif loc.model == 'tesseract':
                roi = self.tesseract_ocr.preprocess_image(roi)
                extractor = self.tesseract_ocr
            elif loc.model == 'easyocr':
                roi = self.easy_ocr.preprocess_image(roi)
                extractor = self.easy_ocr
            else:
                raise ValueError(f"Unsupported OCR model: {loc.model}")

            # Extract text using the respective OCR engine
            extracted_text = extractor.extract_text(roi)
            if loc.model == 'hdr':
                extractor.show_image_with_boxes(roi)

            # convert the extracted text to lowercase and then check to see if the
            # text contains any of the filter keywords (these keywords
            # are part of the *form itself* and should be ignored)
            words = extracted_text.split(' ')
            lower = extracted_text.lower().split(' ')
            result = []
            for i in range(len(words)):
                if lower[i] not in loc.filter_keywords:
                    result.append(words[i])

            extracted_texts[loc.id] = ' '.join(result)
        return extracted_texts
