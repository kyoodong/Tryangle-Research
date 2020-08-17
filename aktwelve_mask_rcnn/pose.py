import cv2
import matplotlib.pyplot as plt

# caffemodel 파일 다운로드
# wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -P pose


image = cv2.imread("images/test6.jpg")
height, width = image.shape[0], image.shape[1]
net = cv2.dnn.readNetFromCaffe("pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", "pose/mpi/pose_iter_160000.caffemodel")
inp = cv2.dnn.blobFromImage(image, 1.0 / 255, (width, height), (0, 0, 0), False, False)
net.setInput(inp)

out = net.forward()

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
"Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
             ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
             ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
             ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

points = list()
threshold = 0.5

for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponging body's part.
    heatMap = out[0, i, :, :]

    # Originally, we try to find all the local maximums. To simplify a sample
    # we just find a global one. However only a single pose at the same time
    # could be detected this way.
    _, conf, _, point = cv2.minMaxLoc(heatMap)
    x = (width * point[0]) / out.shape[3]
    y = (height * point[1]) / out.shape[2]

    # Add a point if it's confidence is higher than threshold.
    print("conf", conf)
    points.append((int(x), int(y)) if conf > threshold else None)

for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert (partFrom in BODY_PARTS)
    assert (partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]
    if points[idFrom] and points[idTo]:
        cv2.line(image, points[idFrom], points[idTo], (255, 74, 0), 3)
        cv2.ellipse(image, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
        cv2.ellipse(image, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
        cv2.putText(image, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


plt.imshow(image)
plt.show()