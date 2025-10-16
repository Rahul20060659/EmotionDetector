import cv2
from fer import FER
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    detector = FER(mtcnn=True)  # mtcnn=True is more accurate (slower). Set False if slow.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(rgb)

        for face in results:
            x, y, w, h = face["box"]
            emotions = face["emotions"]
            top_emotion = max(emotions, key=emotions.get)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, top_emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detector (fer)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
