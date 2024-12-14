import numpy as np
import scanner

line_scanner = scanner.LineScanner(remove_background_method=scanner.REMBG)
contour_scanner = scanner.ContourScanner(remove_background_method=scanner.REMBG)
segmentation_scanner = scanner.DocSegmentationScanner(model_path='./models/model_mbv3_iou_mix_2C_aux_ld.pth')


def scan(image: np.ndarray, method):
    if method == "Line Scanner":
        scanned_image = line_scanner.scan(image)
    elif method == "Contour Scanner":
        scanned_image = contour_scanner.scan(image)
    else:
        scanned_image = segmentation_scanner.scan(image)

    return scanned_image
