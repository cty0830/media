import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "faceset")
IMG_SIZE = (180, 180)

face_db = []
labels = []

faceNamesEN = {
    "0": "JENNIE",
    "1": "TZUYU",
    "2": "TRUMP",
    "3": "CHUNG PEI-JUN"
}

faceNamesCN = {
    "0": "JENNIE",
    "1": "子瑜",
    "2": "川普",
    "3": "鐘沛君"
}

training_data = {
    0: ["JENNIE.jpg", "JENNIE1.jpg"],
    1: ["子瑜.jpg", "子瑜1.jpg"],
    2: ["川普.jpg", "川普1.jpg"],
    3: ["鐘沛君.jpg", "鐘沛君1.jpg"]
}

# 路人門檻
UNKNOWN_THRESHOLD = 70

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def extract_face(gray_img):

    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None, None

    largest_face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    x, y, w, h = largest_face

    face = gray_img[y:y+h, x:x+w]

    face = cv2.equalizeHist(face)

    face = cv2.resize(
        face,
        IMG_SIZE
    )

    return face, (x, y, w, h)


def load_and_preprocess(filename):

    filepath = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(filepath):
        print(f"找不到檔案: {filename}")
        return None

    try:

        with open(filepath, "rb") as f:
            img_array = np.frombuffer(
                f.read(),
                np.uint8
            )

        gray = cv2.imdecode(
            img_array,
            cv2.IMREAD_GRAYSCALE
        )

        if gray is None:
            print(f"無法讀取圖片: {filename}")
            return None

        face, _ = extract_face(gray)

        if face is None:
            print(f"未偵測到人臉: {filename}")
            return None

        return face

    except Exception as e:
        print(f"錯誤: {filename} -> {e}")
        return None

def create_result_image(test_file, name_en, confidence, level):

    image_path = os.path.join(BASE_DIR, test_file)

    img = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:

        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        color = (0, 255, 0)

        if name_en == "UNKNOWN":
            color = (0, 0, 255)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        cv2.putText(
            img,
            name_en,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            img,
            f"{confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.putText(
        img,
        level,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return img

def show_result(test_file, name, confidence, level):

    image_path = os.path.join(
        BASE_DIR,
        test_file
    )

    img = cv2.imdecode(
        np.fromfile(
            image_path,
            dtype=np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:

        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        if name == "路人":
            color = (0, 0, 255)      # 紅色
        else:
            color = (0, 255, 0)      # 綠色

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        cv2.putText(
            img,
            name,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            img,
            f"Distance: {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.rectangle(
        img,
        (10, 10),
        (450, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        f"Result: {level}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        f"Face Recognition - {test_file}",
        img
    )

    print("按任意鍵觀看下一張...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================
# 載入訓練資料
# ==========================

for label, files in training_data.items():

    for file in files:

        img = load_and_preprocess(file)

        if img is not None:

            face_db.append(img)
            labels.append(label)

print(f"\n成功載入 {len(face_db)} 張訓練圖片")

if len(face_db) == 0:
    print("沒有可用訓練資料")
    exit()

# ==========================
# LBPH訓練
# ==========================

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(
    face_db,
    np.array(labels)
)

print("\nLBPH 訓練完成")

# ==========================
# 測試圖片
# ==========================

test_files = [
    "測試1.jpg",
    "測試2.jpg",
    "測試3.jpg",
    "測試4.jpg"
]

result_images = []

print("\n=========================================")

for test_file in test_files:

    print(f"\n測試檔案: {test_file}")

    face = load_and_preprocess(test_file)

    if face is None:
        print("未偵測到人臉")
        continue

    label, confidence = recognizer.predict(face)

    if confidence > UNKNOWN_THRESHOLD:

        name_en = "UNKNOWN"
        name_cn = "路人"
        level = "NOT SIMILAR"

    else:

        name_en = faceNamesEN[str(label)]
        name_cn = faceNamesCN[str(label)]

        if confidence <= 40:
            level = "VERY SIMILAR"
        elif confidence <= 65:
            level = "SIMILAR"
        else:
            level = "ACCEPTABLE"

    print(f"預測人物: {name_cn}")
    print(f"LBPH距離: {confidence:.2f}")
    print(f"判定結果: {level}")

    img = create_result_image(
        test_file,
        name_en,
        confidence,
        level
    )

    result_images.append(
        (test_file, img)
    )

    print("-" * 40)

# ==========================
# 四個視窗同時顯示
# ==========================

positions = [
    (50, 50),
    (900, 50),
    (50, 600),
    (900, 600)
]

for i, (filename, img) in enumerate(result_images):

    if img is None:
        continue

    cv2.namedWindow(
        filename,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        filename,
        800,
        500
    )

    x, y = positions[i]

    cv2.moveWindow(
        filename,
        x,
        y
    )

    cv2.imshow(
        filename,
        img
    )

print("\n按任意鍵關閉所有視窗...")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n辨識完成")