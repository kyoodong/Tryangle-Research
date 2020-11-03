class Guide:
    def __init__(self, guide_id):
        self.guide_id = guide_id

    def __str__(self):
        return "{{'Guide':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()


class LineGuide(Guide):
    def __init__(self, guide_id):
        super(LineGuide, self).__init__(guide_id)

    def __str__(self):
        return "{{'LineGuide':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()


class ObjectGuide(Guide):
    def __init__(self, guide_id, diff_x, diff_y):
        super(ObjectGuide, self).__init__(guide_id)
        self.diff_x = diff_x
        self.diff_y = diff_y

    def __str__(self):
        return "{{'ObjectGuide':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()
