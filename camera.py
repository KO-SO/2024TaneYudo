import cv2
import numpy as np
import gradient_utils
import geometric_calculations
import cut_area
import time
# カメラをキャプチャする
cap = cv2.VideoCapture(0)  # カメラのインデックス（通常は0）

while True:
    start_time=time.time()# 処理の開始時間を記録
    
    ret, frame = cap.read()  # カメラから1フレームを取得
    
    #カメラがない場合の例外処理
    if not ret:
        print("NO CAMERA")
        break
    
    # 画像をリサイズする（200x200にリサイズ）
    #frame = cv2.resize(frame, (200, 200))  # (幅, 高さ)
    #勾配画像の生成
    thinned_gp, thinned_gm = gradient_utils.create_binary_image_from_gradients(frame, 53)
    #線分検出
    gp_lines, gm_lines = geometric_calculations.detect_lines(thinned_gp, thinned_gm, rho=1, theta=np.pi/180, threshold=10, minLineLength=10, maxLineGap=15)
    
    #線分検出の結果チェック
    if (gp_lines is None) or (gm_lines is None):#線分がない場合
        #print("線分検出結果：無し")
        print(0,end="")
        cv2.imshow("image",frame)
        
    else:#線分があった場合
        #四角形の切り出し
        result_image,result_vertex_list = cut_area.make_square(frame, gp_lines, gm_lines)
        #赤コーンの検出結果チェック
        
        if result_vertex_list is None:#条件を満たす線分がない場合
            #print("赤コーン検出結果：無し")
            print(0,end="")
            cv2.imshow("image",frame)
            
        else:#条件を満たす線分があった場合
            #重心の計算
            centroid_image = geometric_calculations.calculate_centroid(frame, result_vertex_list)
            print(1,end="")
            cv2.imshow("image", centroid_image)
            #cv2.destroyAllWindows()
            
    end_time = time.time()  # 処理の終了時間を記録
    elapsed_time = end_time - start_time  # 処理にかかった時間を計算
    
    print(f"   time: {elapsed_time}")
    #'q'を入力するとループを抜けて終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
