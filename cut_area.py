import cv2
import numpy as np


def get_rectangle(gp_line,gm_line):
    vertex_list=None
    lx1, ly1, lx2, ly2 = gp_line[0] # left_line
    rx1, ry1, rx2, ry2 = gm_line[0] # right_line
    if (lx2<=rx1) and (ly2<ly1) and (ry1<ry2):
        vertex_list = [(lx1, ly1), (lx2, ly2), (rx1, ry1), (rx2, ry2)]
    return vertex_list

def crop_rectangle(image,vertex_list):
    # 四角形の領域を切り出す関数
    mask = np.zeros_like(image,dtype=np.uint8) # マスク画像を作成する
    points = np.array(vertex_list, dtype=np.int32) # 頂点の座標を整数に変換する
    cv2.fillConvexPoly(mask, points, [255,255,255]) # マスク画像に四角形を描画する

    
    mask1 = np.zeros_like(image,dtype=np.uint8) # マスク画像を作成する
    cv2.fillConvexPoly(mask1, points, (1,0,0)) # マスク画像に四角形を描画する
    cut_rectangle_area = np.sum(mask1)

    cropped_image = cv2.bitwise_and(image,mask) # 元の画像とマスク画像をAND演算する

    return cropped_image, cut_rectangle_area

def get_ab_sum(cropped_image):
    lab_image=cv2.cvtColor(cropped_image,cv2.COLOR_BGR2Lab)
    a_channel=lab_image[:,:,1]
    b_channel=lab_image[:,:,2]
    #cv2.imshow("a_channel",a_channel)
    a_mask=(a_channel>169) & (a_channel<210)
    b_mask=(b_channel>144) & (b_channel<195)
    or_mask=np.logical_or(a_mask,b_mask)
    
    sum_area=np.sum(or_mask)
    return sum_area

def make_square(image,gp_lines,gm_lines):#main
    max_area_ratio=0
    max_sum_area=0
    max_vertex_list=None
    max_image=None
    count=0
    
    for gp_line in gp_lines:
        for gm_line in (gm_lines):
            vertex_list=get_rectangle(gp_line,gm_line)
            if vertex_list is None:
                continue
            cropped_image, cut_rectangle_area = crop_rectangle(image,vertex_list)
            sum_area=get_ab_sum(cropped_image)

            area_ratio=sum_area/cut_rectangle_area
            # if max_area_ratio<area_ratio:
            #     max_area_ratio=area_ratio
            #     max_vertex_list=vertex_list
            #     max_image=cropped_image
            #print(area_ratio*100,"%")
            if area_ratio> 0.6 and sum_area>max_sum_area:
                max_area_ratio=area_ratio
                max_sum_area=sum_area
                max_vertex_list=vertex_list
                max_image=cropped_image
            #print(area_ratio*100,"%")
            count+=1
    #print(f"count={count},Recognition rate ={max_area_ratio*100} %")
    return max_image,max_vertex_list
            
            