# 概要
* OpenCVを用いた画像認識で股間にリアルタイムでモザイクをかけるアート作品作成した
* 本手法は以前Kinectを使って作成したものと違いRGBカメラのみで実現できるところに優位性がある
* Kinectと同様に人体のボーンを取れるOpenPoseはハイパフォーマンスのマシンを必要とするが本手法はOpenCVの顔認識機能のみで実装されており低スペックなマシンでも実行できる
* 小難しく書いているけど要は「チンチンにリアルタイムにモザイクをかけたよ」という話である

# 従来手法
2010年、マイクロソフトからKinectという画期的なデバイスが発売された。
KinectはRGBカメラと距離カメラから、複数の人間を認識して体の主要な部位の座標(ボーン)を取得することができる。
このデバイスの登場で、ラップトップでもボーンを取得するのが用意になり、Processingや各種言語で利用できた。

このKinectのボーンを取得できる機能を利用して、2014年に股間にリアルタイムでモザイクをかけるというメディアアート作品を発表した(あくまでアートである)
[realtime_crotch_mozaiq](https://github.com/konyu/realtime_crotch_mozaiq)

前述の通りKinectは体の関節などの各部位の点を取得できる。
Kinectで股間領域を特定する場合、股間そのもの位置は取得できないためヘソの位置、両足の付け根付近の位置から股間領域を計算しモザイク処理をした。

Kinectを用いた股間領域特定手法の概念図
![](https://raw.githubusercontent.com/konyu/detection_mans_private_area/master/img/kinect_ver.png)


# 今回提案するOpenCVで顔認識を用いた手法
残念なことにKinect自体はディスコンになってしまった。画像のみでボーンを取得できるものライブラリが現れた。今年2017はRGBカメラとディープラーニングを用いて人間の体の座標を取得できる
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)というものが発表された。
OpenPoseは高性能であるが、マシンパワーを必要とするし、環境構築に時間的、費用的なコストがかかる。

そこでもっとOpenPoseを使うこと無く、股間を位置を推定しモザイクをかける方法を考案した。


## システム構成
* OpenCV3
* Python3
* MacBook Pro搭載のフロントカメラ

OpenCVのRuby用のライブラリはOpenCVは2系までしか対応しておらず。Processingも2系しか対応していなかった。
こういうのは

## 提案手法詳細

従来からあるOpenCVの顔認識機能はかなり低コストに画面上の顔領域のピクセル位置を取得することができる。

また顔の領域の鉛直線下方にあるのはあらゆる人種においてもで自明である。しかし、ある程度の位置は絞れても、顔の大きさと胴体の長さは人種、年齢などによってブレが大きい。

ある程度の汎用性を持たせつつ、股間領域を推定する方法は無いだろうか？
先日東京大学暦本研究室のオープンハウスを見学した際に、ある大学院生の研究の実装方法がこの問題を解決するヒントとなった。

その実装方法とは人間が正面を向いているか下を向いているの判定を頭の領域の黒い領域の割合で決めているというものだった。

「これだ！」天啓が降りてきた。

少なくとも多くのアジア人の大人が持っている股間付近の黒いもの、そう**陰毛**である。

つまり顔の位置から鉛直方向にある程度下の領域から、黒っぽい色を探し出し、そこから下にある一定領域が股間領域を推定できる。

画像からある色の領域を二値化するコマンドがあるので容易である。
黒い領域を取り出した際に、下記画像のように白い領域の用に白(255)となる


黒のフリースを北胸から上の画像
![](https://raw.githubusercontent.com/konyu/detection_mans_private_area/master/img/color_mask.png)

### プログラムの概要
このような処理をループ処理することにより股間にモザイクをかける処理を実装している

1. カメラから画像を取得する
1. 画像から顔認識する
1. 画像から黒い領域を取得する
1. 顔の位置から下方向にある程度幅をもたせた検索領域を部分領域(Region of Interest)で取得する
1. 探索領域にある黒い領域のピクセル位置から陰毛領域を取得する
1. 陰毛領域から一定サイズの四角い領域をモザイク処理する


OpenCVの顔認識機能を用いた股間領域特定手法の概念図
![](https://raw.githubusercontent.com/konyu/detection_mans_private_area/master/img/faceditection_ver.png)


## デモ動画
リアルに股間丸出しででも動画を取ったけれども、あまりにもリアルだったので、
黒いモバイルバッテリーで試したデモ動画をアップした(リアルバージョンは実際にオレに会ったら見せるよ)

[![](http://i.ytimg.com/vi/yvKvqHBRzsY/sddefault.jpg)](https://youtu.be/yvKvqHBRzsY)
