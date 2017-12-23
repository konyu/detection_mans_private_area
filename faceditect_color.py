import numpy as np
import cv2
import time
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300) # カメラ画像の縦幅を720に設定


# cascade_path =  "./data/haarcascade_frontalface_alt_tree.xml"
cascade_path =  "./data/haarcascade_frontalface_default.xml"

# cascade_path =  "./data/haarcascade_fullbody.xml"
# cascade_path =  "./data/haarcascade_lowerbody.xml"

cascade = cv2.CascadeClassifier(cascade_path)
#顔認識の枠の色
color = (255, 0, 0)

lower_yellow = np.array([0, 0, 0])
upper_yellow = np.array([50, 50, 50])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 指定した色に基づいたマスク画像の生成
    # 対象領域が白くなっている
    img_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # フレーム画像とマスク画像の共通の領域を抽出する。
    # img_color = cv2.bitwise_and(frame, frame, mask=img_mask)
    # cv2.imshow("SHOW COLOR IMAGE", img_mask)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
    # 顔の領域は1つとする
    if len(faces) < 1:
        # TODO とりあえず顔判別できなかったら終了処理する。メソッドかした方がいい
        cv2.imshow('frame',frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue

    face = faces[0]
    # 囲う四角の左上の座標
    coordinates = tuple(face[0:2])
    # (囲う四角の横の長さ, 囲う四角の縦の長さ)
    length = tuple(face[0:2] + face[2:4])
    # 四角で囲う処理
    # cv2.rectangle(frame, coordinates, length, color, thickness=3)
    # cv2.rectangle(frame, coordinates, length, color, thickness=2)

    # TODO 股間の検索領域を絞る
    # とりあえず顔の領域
    x = face[0]
    w = x + face[2]
    y = face[1]
    h = y + face[3]

    # 検索範囲を下の方にする下駄
    # y_d = 0
    # 顔の幅に係数をかけて算出する
    y_d = int(face[2] * 1.0)
    # import pdb; pdb.set_trace()

    # 枠がFrameから外れる可能性があった場合は落ちる？
    searched_frame = img_mask[(y + y_d):(h + y_d), x:w]
    # 検索範囲を四角で囲う処理
    cv2.rectangle(frame, (x, (y + y_d)), (w, (h + y_d)), color, thickness=1)
    # cv2.rect(gray, (x, (y + y_d)), (w, (h + y_d)), color, thickness=3)

    # searched_frame = img_mask[(y + y_d):(h + y_d), x:w]

    t_x = 0
    t_y = 0

    for i, row in enumerate(searched_frame):
        for j, cell in enumerate(row):
            if cell == 255:
                t_y = i
                t_x = j
                break
        else:
            continue
        break
    # import pdb; pdb.set_trace()
    #frame[行 start:行 end, 列 strat:列 end]
    # frame[50:100,500:700] = [255, 255, 255]
    # frame[0:h-y,0:w-x] = searched_frame

    # frame[(x + t_x + x_d):(x + t_x + x_d + 10), (y + t_y):(y + t_y + 10)] = [255, 255, 255]
    # チンコをマスクするサイズ
    tnk_w = int(face[2] * 0.9)

    #モザイク処理
    mosaicWidth = 25
    mosaicHeight = 25
    for j in range(y, h, mosaicHeight):
        for i in range(x, w, mosaicWidth):
            # TODO ウインドウサイズを超えたらエラーになるのを避ける必要はあるか？
            tmp_color = frame[j, i]
            # frame[j:(j + mosaicHeight), i:(i + mosaicWidth)] = tmp_color
            frame[j:(j + mosaicHeight), i:(i + mosaicWidth)] = tmp_color



    # gray[(y  + y_d + t_y):(y + y_d + t_y + tnk_w), (x + t_x):(x + t_x + tnk_w)] = 255

    # とりあえず全部白くしておく
    # gray[(y  + y_d + t_y):(y + y_d + t_y + tnk_w), (x + t_x):(x + t_x + tnk_w)] = 255
    # frame[(y  + y_d + t_y):(y + y_d + t_y + 50), (x + t_x):(x + t_x + 50)] = [255, 255, 255]

    # frame[(x):(w), (y):(h)] = [255, 255, 255]

    #　TODO img_maskのある領域からから白いところを探す

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    cv2.imshow('frame',frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
