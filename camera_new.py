import cv2
import numpy as np
import time

# 画像を表すクラス
class Image:
    # コンストラクタ
    def __init__(self,frame):
        self.frame = frame # 画像データ
        self.lab_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2Lab)
        self.thinned_gp = None # 勾配画像（正）
        self.thinned_gm = None # 勾配画像（負）
        self.gp_lines = None # 線分（正）
        self.gm_lines = None # 線分（負）
        self.result_image = None # 四角形の切り出し画像
        self.result_vertex_list = None # 四角形の頂点リスト
        self.centroid_x_coord=-1 # 重心の座標
        self.centroid_image = frame # 重心の位置に縦線を引いたもの
    
    # 勾配画像の生成メソッド
    def create_binary_image_from_gradients(self, tg=50):
        # a*チャンネルの横方向のSobel勾配を計算
        ga = cv2.Sobel(self.lab_image[:, :, 1], cv2.CV_64F, 1, 0, ksize=3) # type: ignore
        gb = cv2.Sobel(self.lab_image[:, :, 2], cv2.CV_64F, 1, 0, ksize=3) # type: ignore

        # 勾配の絶対値を計算
        gap = np.maximum(ga, 0)
        gam = np.maximum(-ga, 0)
        gbp = np.maximum(gb, 0)
        gbm = np.maximum(-gb, 0)

        # しきい値を適用して2値画像を作成
        gp = ((gap + gbp) > tg).astype(np.uint8)
        gm = ((gam + gbm) > tg).astype(np.uint8)

        self.thinned_gp, self.thinned_gm = gp,gm
        
    # 線分検出メソッド
    def detect_lines(self, rho, theta, threshold, minLineLength, maxLineGap):
        # 8ビット符号なし整数型に変換
        gp = cv2.convertScaleAbs(self.thinned_gp)
        gm = cv2.convertScaleAbs(self.thinned_gm)
        # Hough変換を適用
        self.gp_lines = cv2.HoughLinesP(gp, rho, theta, threshold, minLineLength, maxLineGap)
        self.gm_lines= cv2.HoughLinesP(gm, rho, theta, threshold, minLineLength, maxLineGap)
    
    # 四角形の切り出しメソッド
    def cut_red_cone_area(self,a_max,a_min,b_max,b_min,threshold):

                max_sum_area=0
                max_vertex_list=None
                max_image=None
                
                for gp_line in self.gp_lines:
                    for gm_line in self.gm_lines:
                        #線分の組み合わせ選定
                        vertex_list=None
                        lx1, ly1, lx2, ly2 = gp_line[0] # left_line
                        rx1, ry1, rx2, ry2 = gm_line[0] # right_line
                        if (lx2<=rx1) and (ly2<ly1) and (ry1<ry2):
                            vertex_list = [(lx1, ly1), (lx2, ly2), (rx1, ry1), (rx2, ry2)]
                        if vertex_list is None:
                            continue
                        #線分座標から赤コーンのエリアを切り取る
                        mask = np.zeros_like(self.frame,dtype=np.uint8) # マスク画像を作成する
                        points = np.array(vertex_list, dtype=np.int32) # 頂点の座標を整数に変換する
                        cv2.fillConvexPoly(mask, points, [255,255,255]) # マスク画像に四角形を描画する
                        mask1 = np.zeros_like(self.frame,dtype=np.uint8) # マスク画像を作成する
                        cv2.fillConvexPoly(mask1, points, (1,0,0)) # マスク画像に四角形を描画する
                        cut_rectangle_area = np.sum(mask1)
                        cropped_image = cv2.bitwise_and(self.frame,mask) # 元の画像とマスク画像をAND演算する
                        
                        a_channel=self.lab_image[:,:,1]
                        b_channel=self.lab_image[:,:,2]
                        a_mask=(a_channel>a_min) & (a_channel<a_max)
                        b_mask=(b_channel>b_min) & (b_channel<b_max)
                        or_mask=np.logical_or(a_mask,b_mask)
                        sum_area=np.sum(or_mask)
                        
                        area_ratio=sum_area/cut_rectangle_area
                        
                        #赤色の比率だけでは，小さい四角形を切り取ってしまうため，面積の大きさも考慮する
                        if area_ratio> threshold and sum_area>max_sum_area:
                            max_sum_area=sum_area
                            max_vertex_list=vertex_list
                            max_image=cropped_image


                self.result_image=max_image
                self.result_vertex_list=max_vertex_list
    
    # 重心の計算メソッド
    def calculate_centroid_x_coord(self):
        # result_vertex_listは4つの頂点の座標を含むリストまたは配列
        # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]のような形式です

        # 各頂点のx座標とy座標をそれぞれ取り出す
        x_coords = [point[0] for point in self.result_vertex_list]

        # x座標の平均を計算して重心座標を得る
        self.centroid_x_coord=np.mean(x_coords)
        #求まったx座標に縦線を描画
        self.centroid_image[:,int(self.centroid_x_coord),1]=255
        self.centroid_image[:,int(self.centroid_x_coord)-1,1]=255
        self.centroid_image[:,int(self.centroid_x_coord)+1,1]=255
        
    #赤コーンの探索
    def image_process(self):
        self.create_binary_image_from_gradients()
        self.detect_lines(rho=1, theta=np.pi / 180, threshold=50, minLineLength=10, maxLineGap=10)
        
        if (self.gp_lines is not None) and (self.gm_lines is not None):
            self.cut_red_cone_area(a_max = 210, a_min = 169, b_max = 195, b_min = 144, threshold = 0.6)
            if self.result_vertex_list is not None:
                self.calculate_centroid_x_coord()
        
#カメラ呼び出し
class Camera:
    def __init__(self):
        # カメラをキャプチャする
        self.cap = cv2.VideoCapture(0)  # カメラのインデックス（通常は0）
        
    def find_red_cone(self):
        
        while True:
            
            ret, frame = self.cap.read()  # カメラから1フレームを取得
            #カメラがない場合の例外処理
            if not ret:
                print("NO CAMERA")
                break
            
            img=Image(frame)
            
            start_time=time.time()# 処理の開始時間を記録
            img.image_process()#画像処理
            end_time = time.time()  # 処理の終了時間を記録
            elapsed_time = end_time - start_time  # 処理にかかった時間を計算
            print(f"time: {elapsed_time}")
            cv2.imshow("frame",img.centroid_image)
            #'q'を入力するとループを抜けて終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 後処理
        self.cap.release()
        cv2.destroyAllWindows()



def main():
    camera=Camera()
    camera.find_red_cone()
    
if __name__ == "__main__":
    main()


        
        
        
        

                
