# bu uzgarish 1-local repodan
import cv2
import numpy as np
# Bu uzgarish clone repodan
#yangi ozgarishlar bor buyerda
def segment_image_kmeans(image_path, K=99):
    # Tasvirni o'qish
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))  # Hajmni kichraytirish ixtiyoriy
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Piksellarni 1D formatga o‘tkazish (K-means uchun)
    pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)

    # K-means parametrlari
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Klaster markazlari (ranglar)
    centers = np.uint8(centers)

    # Har pikselga mos klaster markazini berish
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_rgb.shape)

    # Natijalarni ko‘rsatish
    cv2.imshow('Asl rasm', img)
    cv2.imshow(f'{K} segmentli rasm', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Natijani saqlash
    cv2.imwrite('segmentlangan_rasm.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

# Dastur ishga tushirish
segment_image_kmeans('Segment.jpg', K=3)