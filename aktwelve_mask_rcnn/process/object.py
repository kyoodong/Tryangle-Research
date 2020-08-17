
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
    def __init__(self, object, pose, pose_class, cropped_image):
        super(Human, self).__init__(object.roi, object.mask, object.clazz, object.score, object.center_point)
        self.pose = pose
        self.pose_class = pose_class
        self.cropped_image = cropped_image