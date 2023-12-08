import cv2
import numpy as np
import get_lab_utils

path='image/1.jpg'
image = cv2.imread(path)

lab_image=cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
L,a,b=get_lab_utils.get_lab(lab_image)
get_lab_utils.create_customized_plot(b,a,"b","a",)


cv2.waitKey(0)
cv2.destroyAllWindows()