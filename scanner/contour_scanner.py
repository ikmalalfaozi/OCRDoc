import cv2
import numpy as np
import imutils
import scanner.utils as utils


class ContourScanner(object):
    def __init__(self, remove_background_method=None):
        """
        Args:
            remove_background_method (string): The methode used to remove the background
                of image (None, GrubCut, RemBg)
        """
        self.method = remove_background_method

    def scan(self, image: str | np.ndarray, shape=None, interactive=False):
        # read image
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # resize image to workable size
        height = 720
        ratio = img.shape[0] / height
        rescaled_img = imutils.resize(img, height=height)

        # get the corners of the document
        corners = self.get_corners(rescaled_img)

        if interactive:
            corners = utils.interactive_get_contour(corners, cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2RGB))
            corners = corners.astype(np.float32)

        # apply the perspective transformation
        warped = utils.four_point_transform(img, corners * ratio, shape)

        return warped

    def get_corners(self, img):
        """
        Returns a numpy array of shape (4, 2) containing the vertices of the four corners
        of the document in the image. If no corners were found, it returns the original
        four corners.
        """
        # remove background
        img = utils.remove_background(img, self.method)
        # Get height and width of image
        IM_HEIGHT, IM_WIDTH = img.shape[:2]

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # Edge Detection.
        canny = cv2.Canny(gray, 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Finding contours for the detected edges.
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Keeping only the largest detected contour.
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Detecting Edges through Contour approximation.
        # Loop over the contours.
        corners = None
        for c in page:
            # Approximate the contour.
            epsilon = 0.02 * cv2.arcLength(c, True)
            _corners = cv2.approxPolyDP(c, epsilon, True)
            # If our approximated contour has four points.
            if len(_corners) == 4:
                corners = _corners
                break

        if corners is None:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            corners = np.array([TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT], dtype='float32')
        else:
            corners = corners.reshape(4, 2)
            corners = utils.order_points(corners)

        return corners
