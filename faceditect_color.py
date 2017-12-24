import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450) # カメラ画像の縦幅を720に設定

cascade_path =  "./data/haarcascade_frontalface_default.xml"

cascade = cv2.CascadeClassifier(cascade_path)
#顔認識の枠の色
color = (255, 0, 0)

lower_black = np.array([0, 0, 0])
upper_black = np.array([50, 50, 50])
# upper_black = np.array([5, 5, 5])

while(True):
    ret, frame = cap.read()

    # 左右反転
    frame = cv2.flip(frame, 1)
    window_height, window_width = frame.shape[:2]

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 指定した色に基づいたマスク画像の生成
    # 対象領域が白くなっている
    img_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 顔認識
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
    # 顔の領域は1つとする
    if len(faces) < 1:
        # とりあえず顔判別できなかったら終了処理する。メソッドかした方がいい
        # ここで表示しちゃうと全力でチンコが映るのでコメント化
        # cv2.imshow('frame',frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue

    face = faces[0]
    # 囲う四角の左上の座標
    coordinates = tuple(face[0:2])
    # (囲う四角の横の長さ, 囲う四角の縦の長さ)
    length = tuple(face[0:2] + face[2:4])
    # 四角で囲う処理
    # cv2.rectangle(frame, coordinates, length, color, thickness=2)

    # 顔の領域
    x = face[0]
    w = x + face[2]
    y = face[1]
    h = y + face[3]

    # 検索範囲を下の方にする下駄
    # 顔の幅に係数をかけて算出する
    # y_d = int(face[2] * 0)
    y_d = int(face[2] * 3.5)

    searched_frame = img_mask[(y + y_d):(h + y_d), x:w]
    # 検索範囲を四角で囲う処理
    # cv2.rectangle(frame, (x, (y + y_d)), (w, (h + y_d)), color, thickness=1)

    t_x = -1
    t_y = -1

    for i, row in enumerate(searched_frame):
        for j, cell in enumerate(row):
            if cell == 255:
                t_y = i
                t_x = j
                break
        else:
            continue
        break

    #frame[行 start:行 end, 列 strat:列 end]

    # Tnkをマスクするサイズ
    tnk_w = int(face[2] * 0.9)
    tnk_h = int(face[2] * 1.0)

    #モザイク処理
    mosaicWidth = 25
    mosaicHeight = 25
    if t_x > 0 and t_y > 0:
        for j in range(y + y_d + t_y, y + y_d + t_y + tnk_h, mosaicHeight):
            for i in range(x + t_x, x + t_x + tnk_w, mosaicWidth):
                # ウインドウサイズを超えたらエラーになるのを避ける
                if (j >= window_height or i >= window_width):
                      continue

                if (j + mosaicHeight)> window_height:
                    mosaicHeight = window_height - j


                tmp_color = frame[j, i]
                frame[j:(j + mosaicHeight), i:(i + mosaicWidth)] = tmp_color
    # 表示
    cv2.imshow('frame',frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
