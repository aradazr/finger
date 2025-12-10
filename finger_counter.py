import cv2
import mediapipe as mp
import math
from collections import deque
from datetime import datetime

# بریم سر اصل مطلب: کتابخونه‌های مدیاپایپ برای دست و صورت
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# شماره‌ی لندمارک نوک انگشت‌ها (از شست تا کوچیکه) توی MediaPipe
TIP_IDS = [4, 8, 12, 16, 20]

# یه صف کوچیک برای ذخیره رویدادها (اگه بعداً خواستیم بفرستیم سمت API)
EVENT_BUFFER = deque(maxlen=50)

# گفتار متن اختیاری؛ اگه pyttsx3 نصب نباشه، صدا غیرفعاله
try:
    import pyttsx3

    TTS_ENGINE = pyttsx3.init()
except Exception:
    TTS_ENGINE = None


def count_fingers(hand_landmarks, handedness_label: str) -> int:
    """
    این تابع می‌شماره چند تا انگشت بالا رفته.
    برای شست محور x مهمه (چون تصویر آینه‌ای می‌شه)،
    برای بقیه انگشت‌ها محور y (نوک بالاتر از مفصل دوم باشه).
    """
    landmarks = hand_landmarks.landmark
    fingers_up = 0

    # Thumb: x comparison depends on handedness (because of mirroring)
    if handedness_label == "Right":
        fingers_up += int(landmarks[TIP_IDS[0]].x < landmarks[TIP_IDS[0] - 1].x)
    else:  # Left
        fingers_up += int(landmarks[TIP_IDS[0]].x > landmarks[TIP_IDS[0] - 1].x)

    # Four other fingers: tip higher (smaller y) than the PIP joint
    for tip_id in TIP_IDS[1:]:
        fingers_up += int(landmarks[tip_id].y < landmarks[tip_id - 2].y)

    return fingers_up


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def detect_emotion(face_landmarks) -> str:
    """
    حدس ساده‌ی حالت چهره بدون مدل یادگیری:
    با نسبت باز و بسته بودن دهان و بالا/پایین بودن گوشه لب،
    سه حالت Happy / Sad / Neutral رو تشخیص می‌ده.
    """
    lm = face_landmarks.landmark
    # Mouth reference points from FaceMesh
    left_corner, right_corner = lm[61], lm[291]
    upper_lip, lower_lip = lm[13], lm[14]

    mouth_width = _dist(left_corner, right_corner)
    mouth_height = _dist(upper_lip, lower_lip)
    ratio = mouth_height / (mouth_width + 1e-6)

    mouth_center_y = (upper_lip.y + lower_lip.y) / 2.0
    corners_avg_y = (left_corner.y + right_corner.y) / 2.0
    smile_curve = mouth_center_y - corners_avg_y  # positive if corners are higher

    # Heuristic thresholds tuned for normalized coords
    if ratio > 0.32 or smile_curve > 0.008:
        return "Happy"
    if smile_curve < -0.008:
        return "Sad"
    return "Neutral"


def classify_gesture(hand_landmarks, handedness_label: str, fingers_up: int) -> str:
    """
    تشخیص ژست‌های ساده بر اساس لندمارک و تعداد انگشت:
    OK، Peace/V، Like (شست بالا)، Dislike (شست پایین).
    """
    lm = hand_landmarks.landmark

    # Thumb direction: compare tip/prox x with wrist
    wrist = lm[0]
    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]

    # Vector for thumb to wrist to decide up/down in image space (y)
    thumb_up = thumb_tip.y < wrist.y - 0.05
    thumb_down = thumb_tip.y > wrist.y + 0.05

    # OK gesture: thumb/index tips near each other, other fingers down
    ok_dist = _dist(thumb_tip, index_tip)
    ok_close = ok_dist < 0.05
    if ok_close and fingers_up <= 2:
        return "OK"

    # Peace/V: two fingers up (index, middle), others down
    if fingers_up == 2 and lm[8].y < lm[6].y and lm[12].y < lm[10].y:
        return "Peace"

    # Like / Dislike: thumb up or down, other fingers folded
    if fingers_up == 1 and thumb_up:
        return "Like"
    if fingers_up == 1 and thumb_down:
        return "Dislike"

    return ""


def speak(text: str) -> None:
    """اگه موتور TTS آماده بود، همین متن رو می‌خونه."""
    if not TTS_ENGINE:
        return
    try:
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()
    except Exception:
        pass


def log_event(kind: str, payload: str) -> None:
    """رویداد رو تو بافر می‌ریزیم؛ بعداً می‌تونیم بفرستیم سمت وب‌سوکت/API."""
    EVENT_BUFFER.append(
        {"time": datetime.utcnow().isoformat(), "kind": kind, "payload": payload}
    )


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("وب‌کم باز نشد!")
        return

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # تصویر رو آینه‌ای می‌کنیم که کاربر گیج نشه
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb)
            face_result = face_mesh.process(rgb)

            # یه هشدار ساده برای نور کم
            brightness = frame.mean()
            if brightness < 40:
                cv2.putText(
                    frame,
                    "Low light: increase lighting",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # پردازش دست‌ها
            total_fingers = 0
            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hand_result.multi_hand_landmarks, hand_result.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Right" or "Left"
                    count = count_fingers(hand_landmarks, label)
                    total_fingers += count
                    gesture = classify_gesture(hand_landmarks, label, count)

                    # لندمارک‌ها و اتصال‌ها رو رسم می‌کنیم
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # یه باکس ساده دور دست برای نوشتن متن
                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]
                    h, w, _ = frame.shape
                    x_min, x_max = int(min(x_vals) * w), int(max(x_vals) * w)
                    y_min, y_max = int(min(y_vals) * h), int(max(y_vals) * h)
                    cv2.rectangle(
                        frame,
                        (x_min - 10, y_min - 10),
                        (x_max + 10, y_max + 10),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{label}: {count}" + (f" | {gesture}" if gesture else ""),
                        (x_min, y_min - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    if gesture:
                        log_event("gesture", f"{label}:{gesture}")
                        speak(gesture)

            # نمایش مجموع انگشت‌ها
            if total_fingers:
                cv2.putText(
                    frame,
                    f"Total fingers: {total_fingers}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

            # پردازش چهره برای Emotion
            if face_result.multi_face_landmarks:
                face_landmarks = face_result.multi_face_landmarks[0]
                emotion = detect_emotion(face_landmarks)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 255), thickness=1, circle_radius=1
                    ),
                )

                x_vals = [lm.x for lm in face_landmarks.landmark]
                y_vals = [lm.y for lm in face_landmarks.landmark]
                h, w, _ = frame.shape
                x_min, x_max = int(min(x_vals) * w), int(max(x_vals) * w)
                y_min, y_max = int(min(y_vals) * h), int(max(y_vals) * h)
                cv2.rectangle(
                    frame,
                    (x_min - 10, y_min - 10),
                    (x_max + 10, y_max + 10),
                    (255, 215, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (x_min, max(20, y_min - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 215, 255),
                    2,
                )

                # Simple AR emoji overlay near forehead
                emoji = {"Happy": ":)", "Sad": ":(", "Neutral": ":|"}[emotion]
                cv2.putText(
                    frame,
                    emoji,
                    (x_min + (x_max - x_min) // 2, y_min - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 200, 255),
                    3,
                )

                log_event("emotion", emotion)
                speak(emotion)

            cv2.imshow("Finger Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

