import numpy as np


def checkBoundary(box, image_size):
    min_x,min_y,max_x,max_y=box
    img_height, img_width = image_size
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(img_width, max_x), min(img_height, max_y)
    return (min_x,min_y,max_x,max_y)

def getEnclosingBBox(boxesLst):
    boxes_np = np.array(boxesLst)
    # min_x = np.min(boxes_np[:,0],initial=np.inf)
    # min_y = np.min(boxes_np[:,1],initial=np.inf)
    # max_x = np.max(boxes_np[:,2], initial=-np.inf)
    # max_y = np.max(boxes_np[:,3], initial=-np.inf)

    # ====get mode instead of min and max to get tide bboxes
    min_x = np.min(boxes_np[:, 0], initial=np.inf)
    min_y = np.mean(boxes_np[:, 1])
    max_x = np.max(boxes_np[:, 2], initial=-np.inf)
    max_y = np.mean(boxes_np[:, 3])
    return (int(min_x),int(min_y),int(max_x),int(max_y))
