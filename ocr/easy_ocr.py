import cv2
import easyocr
import imutils
import matplotlib.pyplot as plt


class EasyOCR:
    def __init__(self, languages=['en']):
        """
        Initialize the EasyOCR class.

        Args:
            languages: list of str, languages to recognize (default is English)
        """
        self.reader = easyocr.Reader(languages)

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
            img = cv2.imread(image)
        else:
            # Use the image directly if it's already in numpy array format
            img = image

        # Convert the image to grayscale if it is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def extract_text(self, img):
        """
        Extract text from a preprocessed image using EasyOCR.

        Args:
            img: 2-D array, preprocessed image

        Returns:
            str, extracted text
        """
        # Use EasyOCR to extract text from the image
        results = self.reader.readtext(img, detail=0)
        text = '\n'.join(results)
        return text

    def extract_text_with_boxes(self, img):
        """
        Extract text with bounding boxes from a preprocessed image using EasyOCR.

        Args:
            img: 2-D array, preprocessed image

        Returns:
            list of dicts, each containing 'text', 'left', 'top', 'width', 'height' of the bounding boxes
        """
        # Use EasyOCR to extract data with bounding boxes
        results = self.reader.readtext(img)

        text_boxes = []
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            left, top = int(top_left[0]), int(top_left[1])
            right, bottom = int(bottom_right[0]), int(bottom_right[1])
            width, height = right - left, bottom - top
            text_boxes.append({
                'text': text,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'probability': prob
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
            (left, top, width, height) = (box['left'], box['top'], box['width'], box['height'])
            img_color = cv2.rectangle(img_color, (left, top), (left + width, top + height), (0, 255, 0), 2)

        plt.imshow(img_color)
        plt.title('OCR Results')
        plt.show()


# Example usage:
# image_path = '../output/ktp.jpg'
# image = cv2.imread(image_path)
# ocr_engine = EasyOCR(languages=['id'])
# preprocessed_img = ocr_engine.preprocess_image(image)
# text = ocr_engine.extract_text(preprocessed_img)
# print(f'Extracted Text: \n{text}')
# text_boxes = ocr_engine.extract_text_with_boxes(preprocessed_img)
# ocr_engine.show_image_with_boxes(preprocessed_img, text_boxes)
