import cv2
import numpy as np
from insightface.app import FaceAnalysis

fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=0, det_size=(640, 640))

img1 = cv2.imread("person1.jpg")
img2 = cv2.imread("person2.jpg")

if img1 is None or img2 is None:
    print("Error: Could not load one or both images.")
    exit()

facesA = fa.get(img1)
facesB = fa.get(img2)

if len(facesA) == 0:
    print("No face detected in person1.jpg")
    exit()

if len(facesB) == 0:
    print("No face detected in person2.jpg")
    exit()

faceA = facesA[0]
faceB = facesB[0]

code1 = faceA.embedding
code2 = faceB.embedding

distance = np.linalg.norm(code1 - code2)

print("Face Distance:", distance)

if distance < 1.0:
    print("These are likely the same person.")
else:
    print("These are likely different people.")

for face in facesA:
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Face", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()