# tennis_openpose

Сайн байна уу энэхүү прожектийг Бакалаврын дипломын ажлын хүрээнд хийв. Эхлээд notebook.ipynb файлыг уншаарай.

Эхлээд conda орчин үүсгэхийг санал болгож байна. Дараагаар хэрэгтэй сангуудаа суулгана.

```
pip install -r requirements.txt
```

# movenet

Movenet моделыг дараах линкээр суулгах боломжтой. Jupyter Notebook-н дээр илүү дэлгэрэнгүй моделуудаа суулгах заавар байгаа.

```
wget -q -O movenet.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
```

# dataset

Өгөгдлийн санг бэлдэхдээ хүссэн тоглогчийн бичлэгийг бэлдэж авах хэрэгтэй. Ойроос цохиж буй бичлэгнүүд байх тусам сайн. Тэдгээрээ videos хавтас үүсгэн хуулна.
Дараагаар гар дээрх

```
W-Serve
S-Idle
L-Backhand
R-Righthand
```

тэмдэглэгээнүүдийг ашиглаж бичлэгнээс цохилтуудыг тэмдэглэж авна. Анхааруулга: Mediapipe болон movenet моделууд заавал суусан байх ёстой.

```
python annotator.py path/to/video --output-dir annotations
```

Үүний дараа

```
python extract_keypoints.py
```

ашиглан бүх өгөгдлүүдээ нэгтгэж annotation үүсгэнэ.

#testing

Notebook ашиглан сургалтаа(сургалт хийсэн model цуг хавсаргагдсан байгаа) хийж дууссан бол

```
python detect_from_cam.py path/to/model.h5 --output output_webcam.mp4
```

ашиглан тестелж камер дээр үзэх боломжтой.

Бичлэгэн дээр үзэх бол

```
python track_and_classify_with_rnn.py path/to/model.h5  path/to/input.mp4 --output output_webcam.mp4
```

Энэхүү прожектийг ашиглаж үзсэнд баярлалаа.
