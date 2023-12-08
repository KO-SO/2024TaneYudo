import cv2
import numpy as np

# 直線を検出して描画
#detected_gp = Hough.detect_and_draw_lines(image.copy(), thinned_gp, rho=1, theta=np.pi/180, threshold=10,minLineLength=10,maxLineGap=15)
#detected_gm = Hough.detect_and_draw_lines(image.copy(), thinned_gm, rho=1, theta=np.pi/180, threshold=100,minLineLength=10,maxLineGap=5)
def detect_and_draw_lines(image, thinned_image, rho, theta, threshold,minLineLength,maxLineGap):
    # 8ビット符号なし整数型に変換
    thinned_image = cv2.convertScaleAbs(thinned_image)

    # Hough変換を適用
    lines = cv2.HoughLinesP(thinned_image, rho, theta, threshold, minLineLength, maxLineGap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] # 線分を描画 
            cv2.line(image, (x1, y1), (x2, y2), (0, 200, 0), 1)

    return image

def draw_lines(image,lines1,lines2):
    
    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0] # 線分を描画 
            cv2.line(image, (x1, y1), (x2, y2), (0, 200, 0), 1)
            image[x1,y1]=(0,200,0)
            image[x2,y2]
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0] # 線分を描画 
            cv2.line(image, (x1, y1), (x2, y2), (200, 0, 0), 1)
            
    return image

def draw_lines_and_circle(image,lines1,lines2):
    # 印をつけるための円を描画
    color1 = (0, 255, 0)  # 青 (BGR形式)
    color2 = (255, 0, 0)  # 緑 (BGR形式)
    radius = 5  # 円の半径
    thickness = -1  # 塗りつぶしのための厚さ（-1は塗りつぶし）
    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0] # 線分を描画 
            cv2.line(image, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.circle(image, (x1, y1), radius, color1, thickness)
            cv2.circle(image, (x2, y2), radius, color2, thickness)
    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0] # 線分を描画 
            cv2.line(image, (x1, y1), (x2, y2), (200, 0, 0), 1)
            cv2.circle(image, (x1, y1), radius, color1, thickness)
            cv2.circle(image, (x2, y2), radius, color2, thickness)
    return image


def detect_lines(gp,gm,rho, theta, threshold,minLineLength,maxLineGap):
    # 8ビット符号なし整数型に変換
    gp = cv2.convertScaleAbs(gp)
    gm = cv2.convertScaleAbs(gm)
    # Hough変換を適用
    lines_gp = cv2.HoughLinesP(gp, rho, theta, threshold, minLineLength, maxLineGap)
    lines_gm= cv2.HoughLinesP(gm, rho, theta, threshold, minLineLength, maxLineGap)

    return lines_gp,lines_gm

def calculate_centroid(image,points):
    # pointsは4つの頂点の座標を含むリストまたは配列
    # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]のような形式です

    # 各頂点のx座標とy座標をそれぞれ取り出す
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # x座標とy座標の平均を計算して重心座標を得る
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    image[:,int(centroid_x),1]=255
    image[:,int(centroid_x)-1,1]=255
    image[:,int(centroid_x)+1,1]=255
    
    return image
