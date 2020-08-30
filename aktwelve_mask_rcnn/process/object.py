
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
    class Part:
        Nose = "Nose"
        # Neck = "Neck"
        RShoulder = "RShoulder"
        RElbow = "RElbow"
        RWrist = "RWrist"
        LShoulder = "LShoulder"
        LElbow = "LElbow"
        LWrist = "LWrist"
        RHip = "RHip"
        RKnee = "RKnee"
        RAnkle = "RAnkle"
        LHip = "LHip"
        LKnee = "LKnee"
        LAnkle = "LAnkle"
        REye = "REye"
        LEye = "LEye"
        REar = "REar"
        LEar = "LEar"

    # BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    #               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8,
    #               "RKnee": 9, "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13,
    #               "REye": 14, "LEye": 15, "REar": 16, "LEar": 17}

    BODY_PARTS = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                  "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8,
                  "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13,
                  "RKnee": 14, "LAnkle": 15, "RAnkle": 16}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

    def __init__(self, obj, pose, pose_class, cropped_image, extended_roi):
        super(Human, self).__init__(obj.roi, obj.mask, obj.clazz, obj.score, obj.center_point)
        self.pose = pose
        self.pose_class = pose_class
        self.cropped_image = cropped_image
        self.extended_roi = extended_roi
