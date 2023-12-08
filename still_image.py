import cv2
import numpy as np
import gradient_utils
import geometric_calculations
import cut_area
import time

path='image/1.jpg'
frame = cv2.imread(path)
# 画像をリサイズする（200x200にリサイズ）
#frame = cv2.resize(frame, (200, 200))  # (幅, 高さ)
cv2.imshow("orign_image",frame)

start_time=time.time()# 処理の開始時間を記録
 
#勾配画像の生成
thinned_gp, thinned_gm = gradient_utils.create_binary_image_from_gradients(frame, 53)
#線分検出
gp_lines, gm_lines = geometric_calculations.detect_lines(thinned_gp, thinned_gm, rho=1, theta=np.pi/180, threshold=10, minLineLength=10, maxLineGap=15)

result=0#赤コーンの判定結果

#線分検出の結果チェック
if (gp_lines is None) or (gm_lines is None):#線分がない場合
    print("線分が検出されませんでした")
    
else:#線分があった場合
    #四角形の切り出し
    result_image,result_vertex_list = cut_area.make_square(frame, gp_lines, gm_lines)
    
    #赤コーンの検出結果チェック
    if result_vertex_list is None:#条件を満たす線分がない場合
        print("赤コーンが検出されませんでした")
        
    else:#条件を満たす線分があった場合
        #重心の計算
        centroid_image = geometric_calculations.calculate_centroid(frame, result_vertex_list)
        result=1
        cv2.imshow("result_image",result_image)
        cv2.imshow("centroid_image", centroid_image)
        
end_time = time.time()  # 処理の終了時間を記録
elapsed_time = end_time - start_time  # 処理にかかった時間を計算

print(f"{result}   time: {elapsed_time}")


cv2.imshow("thinned_gp",thinned_gp*255)
cv2.imshow("thinned_gm",thinned_gm*255)

cv2.waitKey(0)
cv2.destroyAllWindows()
