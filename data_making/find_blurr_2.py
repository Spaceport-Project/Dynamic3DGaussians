import cv2
import numpy as np

def find_blur_regions(image_path, block_size=64, blur_threshold=100.0, min_blocks=1):
    """
    Find a rectangle that covers all regions considered blurry.

    Args:
        block_size: size of each block for local blur measurement.
        blur_threshold: blocks with Laplacian variance < threshold are considered blurry.
        min_blocks: require at least this many blurry blocks; otherwise returns None.

    Returns:
        rect: (x, y, w, h) for bounding box of all blurry blocks, or None if not enough.
        mask_blocks: boolean 2D mask of blurry blocks.
        vis: visualization image with bounding rectangle drawn (if any).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image at path: {}".format(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    h_cropped = (h // block_size) * block_size
    w_cropped = (w // block_size) * block_size
    gray = gray[0:h_cropped, 0:w_cropped]
    vis = img[0:h_cropped, 0:w_cropped].copy()

    h_blocks = h_cropped // block_size
    w_blocks = w_cropped // block_size

    blur_scores = np.zeros((h_blocks, w_blocks), dtype=np.float32)

    for by in range(h_blocks):
        for bx in range(w_blocks):
            y1 = by * block_size
            x1 = bx * block_size
            block = gray[y1:y1 + block_size, x1:x1 + block_size]
            score = cv2.Laplacian(block, cv2.CV_64F).var()
            blur_scores[by, bx] = score

    # True where blurry (low variance)
    # print(blur_scores)
    mask_blocks = blur_scores < blur_threshold

    # No or too few blurry blocks
    print(mask_blocks.sum() , min_blocks)
    if mask_blocks.sum() < min_blocks:
        return None, mask_blocks, vis

    # Coordinates of all blurry blocks
    ys, xs = np.where(mask_blocks)

    # Block-space bounding box
    min_bx, max_bx = xs.min(), xs.max()
    min_by, max_by = ys.min(), ys.max()

    # Convert to pixel coordinates
    x1 = min_bx * block_size
    y1 = min_by * block_size
    x2 = (max_bx + 1) * block_size
    y2 = (max_by + 1) * block_size

    rect = (x1, y1, x2 - x1, y2 - y1)

    # Draw blocks and bounding box
    for by, bx in zip(ys, xs):
        bx_px = bx * block_size
        by_px = by * block_size
        cv2.rectangle(vis, (bx_px, by_px),
                      (bx_px + block_size, by_px + block_size),
                      (0, 255, 255), 1)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return rect, mask_blocks, vis

# Example usage
if __name__ == "__main__":
    rect, mask_blocks, vis = find_blur_regions(
        "../im2_bck.png",
        block_size=64,
        blur_threshold=100.0,  # tune based on your images
        min_blocks=2
    )
    print("Blurry area bounding rect:", rect)
    if rect is not None:
        cv2.imwrite("blurry_regions_vis.jpg", vis)