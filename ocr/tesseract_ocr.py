import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt


class TesseractOCR:
    def __init__(self, tesseract_cmd=None, lang=None):
        """
        Initialize the TesseractOCR class.

        Args:
            tesseract_cmd: str, path to the tesseract executable (optional)
            lang: str, language that will be ocr (optional)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang

    def preprocess_image(self, image):
        """
        Preprocess a single image for OCR.

        Args:
            image: str or 2-D array, path to the image file or input image

        Returns:
            img: 2-D array, preprocessed image
        """
        if isinstance(image, str):
            # Load the image if path is provided
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # Use the image directly if it's already in numpy array format
            img = image

        # Convert the image to grayscale if it is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def extract_text(self, img):
        """
        Extract text from a preprocessed image using Tesseract.

        Args:
            img: 2-D array, preprocessed image

        Returns:
            str, extracted text
        """
        # Use Tesseract to extract text from the image
        text = pytesseract.image_to_string(img, lang=self.lang)
        return text

    def extract_text_with_boxes(self, img):
        """
        Extract text with bounding boxes from a preprocessed image using Tesseract.

        Args:
            img: 2-D array, preprocessed image

        Returns:
            list of dicts, each containing 'text', 'left', 'top', 'width', 'height' of the bounding boxes
        """
        # Use Tesseract to extract data with bounding boxes
        data = pytesseract.image_to_data(img, output_type=Output.DICT, lang=self.lang)

        text_boxes = []
        for i in range(len(data['text'])):
            text_boxes.append({
                'text': data['text'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })

        return text_boxes

    def show_image_with_boxes(self, img, text_boxes):
        """
        Display the image with bounding boxes around recognized text.

        Args:
            img: 2-D array, preprocessed image
            text_boxes: list of dicts, each containing 'text', 'left', 'top', 'width', 'height' of the bounding boxes
        """
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for box in text_boxes:
            (x, y, w, h) = (box['left'], box['top'], box['width'], box['height'])
            img_color = cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imshow(img_color)
        plt.title('OCR Results')
        plt.show()


# Example usage:
# tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Path to the tesseract executable (if needed)
# image_path = '../output/ktp.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image directly if needed
# # ocr = TesseractOCR(tesseract_cmd)
# ocr = TesseractOCR(lang="ind")
# preprocessed_img = ocr.preprocess_image(image)
# text = ocr.extract_text(preprocessed_img)
# print(f'Extracted Text: \n{text}')
# text_boxes = ocr.extract_text_with_boxes(preprocessed_img)
# ocr.show_image_with_boxes(preprocessed_img, text_boxes)
