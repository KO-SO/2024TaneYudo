import cv2
import numpy as np

def create_binary_image_from_gradients(image, tg=50):
    # 画像をCIE L*a*b*色空間に変換
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # a*チャンネルの横方向のSobel勾配を計算
    ga = cv2.Sobel(lab_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3) # type: ignore
    gb = cv2.Sobel(lab_image[:, :, 2], cv2.CV_64F, 1, 0, ksize=3) # type: ignore

    # 勾配の絶対値を計算
    gap = np.maximum(ga, 0)
    gam = np.maximum(-ga, 0)
    gbp = np.maximum(gb, 0)
    gbm = np.maximum(-gb, 0)

    # しきい値を適用して2値画像を作成
    gp = ((gap + gbp) > tg).astype(np.uint8)
    gm = ((gam + gbm) > tg).astype(np.uint8)

    return gp, gm
