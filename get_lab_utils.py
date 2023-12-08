import cv2
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def get_rgb(image):
    # ウィンドウの名前
    window_name = 'Image'

    # 画像内でマウスがクリックされたときに座標と色を表示するコールバック関数
    def get_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = image[y, x]  # (y, x)の位置のピクセル値を取得
            
            print(f"座標 (x, y): ({x}, {y})")
            print(f"RGB 値: {pixel}")
            
    # 画像を表示
    cv2.imshow(window_name, image)

    # マウスクリックのコールバック関数を設定
    cv2.setMouseCallback(window_name, get_color)

    # 画像ウィンドウを表示
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def get_lab(image):
    # ウィンドウの名前
    window_name = 'Image'
    L=[]
    a=[]
    b=[]
    # 画像内でマウスがクリックされたときに座標と色を表示するコールバック関数
    def get_color(event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel = image[y, x]  # (y, x)の位置のピクセル値を取得

            L.append(pixel[0])
            a.append(pixel[1])
            b.append(pixel[2])
            #print(f"座標 (x, y): ({x}, {y})")
            print(f"lab 値: {pixel}")
            
    # 画像を表示
    cv2.imshow(window_name, image)

    # マウスクリックのコールバック関数を設定
    cv2.setMouseCallback(window_name, get_color)

    # 画像ウィンドウを表示
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return L,a,b

# グラフ作成用の関数を定義
def create_customized_plot(x, y, x_label, y_label, title=None, xlim=None, ylim=None, marker_size=6):

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(x, y,"o")
    
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.tick_params(labelsize=14)
    

        
    plt.grid(linestyle='solid', alpha=0.5)
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.figtext(0.5, 0, title, fontsize=14, ha='center')
     # 余白調整
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #plt.subplots_adjust(left=0, right=1, bottom=0, top=1) #この1行を入れる
    #plt.savefig(title, dpi=600)  # 画像として保存
    plt.show()

