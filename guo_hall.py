import cv2
import numpy as np

# thinned_gp=guo_hall.thin_image(gp*255)
# thinned_gm=guo_hall.thin_image(gm*255)
# 細線化
def thin_image(image):
    
    # 細線化(スケルトン化) THINNING_GUOHALL 
    thinned_image   =   cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    return thinned_image