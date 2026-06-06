import cv2

from insightface.app import FaceAnalysis

fa = FaceAnalysis(name = "buffalo_l")
fa.prepare(ctx_id = 0, det_size=(640, 640))

img1 = cv2.imread("person1.jpg")

faces = fa.get(img1)
face = faces[0]

x1, y1,x2,y2 = map(int, face.bbox)
cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()