import cv2
import numpy as np
import imutils
from pylsd.lsd import lsd
import scanner.utils as utils

class LineScanner(object):
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

        CANNY = 84
        IM_HEIGHT, IM_WIDTH = img.shape[:2]

        TOP_RIGHT = (IM_WIDTH, 0)
        BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
        BOTTOM_LEFT = (0, IM_HEIGHT)
        TOP_LEFT = (0, 0)
        default_corners = np.array([TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT], dtype='float32')

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # find edges and mark them in the output map using the Canny algorithm
        edged = cv2.Canny(gray, 0, CANNY)

        # test_corners = self.get_corners(edged)
        lines = lsd(edged)

        if lines.shape[0] < 2:
            return default_corners

        # separate horizontal and vertical lines
        lines = lines.squeeze().astype(np.int32)
        horizontal_lines = lines[np.abs(lines[:, 3] - lines[:, 1]) < np.abs(lines[:, 2] - lines[:, 0])]
        vertical_lines = lines[np.abs(lines[:, 3] - lines[:, 1]) >= np.abs(lines[:, 2] - lines[:, 0])]

        # Draw horizontal lines
        horizontal_lines_canvas = np.zeros_like(gray)
        for x1, y1, x2, y2, _ in horizontal_lines:
            cv2.line(horizontal_lines_canvas, (x1, y1), (x2, y2), 255, 2)

        # Draw vertical lines
        vertical_lines_canvas = np.zeros_like(gray)
        for x1, y1, x2, y2, _ in vertical_lines:
            cv2.line(vertical_lines_canvas, (x1, y1), (x2, y2), 255, 2)

        # Find the 2 longest horizontal lines
        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        horizontal_lines = []
        for contour in contours:
            contour = contour.squeeze()
            min_x = np.min(contour[:, 0], axis=0)
            max_x = np.max(contour[:, 0], axis=0)
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            horizontal_lines.append((min_x, left_y, max_x, right_y))

        # Find the 2 longest vertical lines
        (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        vertical_lines = []
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y = np.amin(contour[:, 1], axis=0)
            max_y = np.amax(contour[:, 1], axis=0)
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            vertical_lines.append((top_x, min_y, bottom_x, max_y))

        # initialize list to store intersection points
        intersection_points = []

        # loop through each pair of horizontal and vertical lines to find intersection points
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                # calculate intersection point
                intersection_x = (h_line[0] * h_line[3] - h_line[1] * h_line[2]) * (v_line[0] - v_line[2]) - \
                                 (h_line[0] - h_line[2]) * (v_line[0] * v_line[3] - v_line[1] * v_line[2])
                intersection_x /= (h_line[0] - h_line[2]) * (v_line[1] - v_line[3]) - \
                                  (h_line[1] - h_line[3]) * (v_line[0] - v_line[2])

                intersection_y = (h_line[0] * h_line[3] - h_line[1] * h_line[2]) * (v_line[1] - v_line[3]) - \
                                 (h_line[1] - h_line[3]) * (v_line[0] * v_line[3] - v_line[1] * v_line[2])
                intersection_y /= (h_line[0] - h_line[2]) * (v_line[1] - v_line[3]) - \
                                  (h_line[1] - h_line[3]) * (v_line[0] - v_line[2])

                # add intersection point to list
                intersection_points.append((intersection_x, intersection_y))

        # convert intersection points to numpy array
        corners = np.array(intersection_points)

        if corners.shape[0] == 4:
            return utils.order_points(corners)

        return default_corners
