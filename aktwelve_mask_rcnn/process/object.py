
class Object:
    def __init__(self, roi, mask, clazz, score, center_point):
        self.roi = roi
        self.mask = mask
        self.clazz = clazz
        self.score = score
        self.center_point = center_point

    def is_person(self):
        return self.clazz == 1


class Human(Object):
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                  "Background": 15}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

    def __init__(self, obj, pose, pose_class, cropped_image):
        super(Human, self).__init__(obj.roi, obj.mask, obj.clazz, obj.score, obj.center_point)
        self.pose = pose
        self.pose_class = pose_class
        self.cropped_image = cropped_image