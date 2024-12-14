import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt


class HDR(object):
    def __init__(self, model_path='models/resnet164.h5'):
        """
        Initialize the HDR class.

        Args:
            model_path: str, path to the trained model file
        """
        self.model = load_model(model_path)

    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction.

        Args:
            image: str or 2-D array, path to the image file or input image

        Returns:
            img: 4-D array, preprocessed image
        """
        if isinstance(image, str):
            # Load the image if path is provided
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # Use the image directly if it's already in numpy array format
            img = image

        # Convert the image to grayscale if it is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert the colors if the background is white and the digit is black
        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        # Add padding to make the image square
        img = make_image_square(img)

        # Resize to 28x28 pixels
        img = cv2.resize(img, (28, 28))

        # Normalize the image
        img = img / 255.0

        # Reshape to match the input shape of the model
        img = img.reshape(1, 28, 28, 1)

        return img

    def predict(self, img):
        """
        Predict the digit from a preprocessed image.

        Args:
            img: 4-D array, preprocessed image

        Returns:
            int, predicted digit
        """
        # Perform inference
        prediction = self.model.predict(img)

        # Convert prediction from one-hot encoding to class label
        predicted_class = np.argmax(prediction, axis=1)[0]

        return predicted_class

    def recognize_multi_digit_image(self, image):
        """
        Recognize multiple handwritten digits from an image.

        Args:
            image: str or 2-D array, path to the image file containing multiple digits

        Returns:
            list of dicts, each containing 'text', 'left', 'top', 'width', 'height' of the bounding box
        """
        if isinstance(image, str):
            # Load the image if path is provided
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # Use the image directly if it's already in numpy array format
            img = image

        # Convert the image to grayscale if it is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Invert the colors if the background is black and the digit is white
        if np.mean(img) < 127:
            img = cv2.bitwise_not(img)

        # Apply Otsu's thresholding with an additional offset so that the digits with
        # thin lines do not separate
        otsu_threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh = cv2.threshold(img, min(otsu_threshold + 15, 255), 255, cv2.THRESH_BINARY_INV)
        # apply open operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Perform segmentation using contours and recognize each digit
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = img.shape
        results = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)

            left = max(x - pad_x, 0)
            top = max(y - pad_y, 0)
            right = min(x + w + pad_x, width)
            bottom = min(y + h + pad_y, height)

            # Filter based on contour size and aspect ratio
            is_number_one = w / h < 0.2  # handle digit "1" issue
            if (10 < w and 10 < h and 0.2 < w / h < 3.0) or is_number_one:
                if is_number_one:
                    pad_x = max(pad_x, 1)
                    left = max(x - 3 * pad_x, 0)
                    right = min(x + w + 3 * pad_x, width)

                # extract roi
                digit_img = thresh[top:bottom, left:right]

                # Preprocess each digit image
                digit_img = make_image_square(digit_img)
                digit_img = cv2.resize(digit_img, (28, 28))
                digit_img = digit_img / 255.0
                digit_img = digit_img.reshape(1, 28, 28, 1)

                # Predict the digit
                predicted_digit = self.predict(digit_img)

                # Store the result
                text_box = {
                    'text': str(predicted_digit),
                    'left': left,
                    'top': top,
                    'width': right - left,
                    'height': bottom - top
                }
                results.append(text_box)

        return results

    def extract_text(self, img):
        """
        Extract text from a preprocessed image using EasyOCR.

        Args:
            img: 2-D array, preprocessed image

        Returns:
            str, extracted text
        """
        results = self.recognize_multi_digit_image(img)
        results = sorted(results, key=lambda x: (x['left'], x['top']))  # Sort based on position
        text = ''.join([result['text'] for result in results])
        return text

    def show_prediction(self, img):
        """
        Display the image with the predicted digit.

        Args:
            img: 2-D array, preprocessed image
        """
        # Perform inference
        predicted_digit = self.predict(img)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'Predicted Digit: {predicted_digit}')
        plt.show()

    def show_image_with_boxes(self, image):
        """
        Display the image with bounding boxes around recognized digits.

        Args:
            image: str or 2-D array, path to the image file containing multiple digits
        """
        results = self.recognize_multi_digit_image(image)
        if isinstance(image, str):
            # Load the image if path is provided
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            # Use the image directly if it's already in numpy array format
            img = image

        # Convert the image to grayscale if it is not already
        if len(img.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_color = cv2.copyMakeBorder(img_color, 20, 0, 0, 20, cv2.BORDER_REPLICATE)

        # Draw bounding boxes
        for box in results:
            left, top, width, height = box['left'], box['top'] + 20, box['width'], box['height']
            img_color = cv2.rectangle(img_color, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(img_color, box['text'], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        plt.imshow(img_color)
        plt.title('Segmented Image with Predicted Digits')
        plt.show()


def make_image_square(img):
    # Add padding to make the image square
    height, width = img.shape
    if height != width:
        # Calculate the padding needed for each side
        size = max(height, width)
        delta_w = size - width
        delta_h = size - height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Add padding
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return img
