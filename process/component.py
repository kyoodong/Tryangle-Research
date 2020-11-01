class Component:
    def __init__(self, id):
        self.id = id
        self.guide_list = list()


class LineComponent(Component):
    def __init__(self, id, line):
        super(LineComponent, self).__init__(id)
        self.line = line

    def __str__(self):
        return "{{'LineComponent':{}}}".format(str(self.__dict__))

    def __repr__(self):
        return self.__str__()


class ObjectComponent(Component):
    def __init__(self, id, object):
        super(ObjectComponent, self).__init__(id)
        self.object = object
