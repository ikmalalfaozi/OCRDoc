import cv2
import numpy as np
import scanner.constants as constants
from rembg import remove
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import scanner.polygon_interacter as poly_i


def remove_background(img, method):
    # Add padding to the image
    padding_size = 24
    img = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

    if method == constants.GRAB_CUT:
        # GrabCut to get rid of the background.
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

    elif method == constants.REMBG:
        img = remove(img)

    img = img[24:-24, 24:-24]

    return img


def interactive_get_contour(corners, image):
    poly = Polygon(corners, animated=True, fill=False, color="yellow", linewidth=1.0)
    fig, ax = plt.subplots()
    ax.add_patch(poly)
    ax.set_title(('Drag the corners of the box to the corners of the document. \n'
                  'Close the window when finished.'))
    p = poly_i.PolygonInteractor(ax, poly)
    plt.imshow(image)
    plt.show()

    new_points = p.get_poly_points()[:4]
    new_points = np.array([[p] for p in new_points], dtype="int32")
    return new_points.reshape(4, 2)


def order_points(pts):
    """Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect


def four_point_transform(image, pts, shape=None):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = pts

    if shape is None:
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")
    else:
        height, width = shape
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.warpPerspective(image, M, (width, height))

    # return the warped image
    return warped

