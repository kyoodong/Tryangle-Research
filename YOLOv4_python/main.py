import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from imageClassifier import  Classifier


def main():
    classifier = Classifier()
    object_list = classifier.get_object_list('./res/test')

    print(object_list)

    for list in object_list:
        print(list)

if __name__ == '__main__':
    main()
