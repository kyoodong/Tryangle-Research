class CocoInfo:
    def __init__(self, description, url, version, year, contributor, date_created):
        self.description = description
        self.url = url
        self.version = version
        self.year = year
        self.contributor = contributor
        self.date_created = date_created

    def __str__(self):
        return '{{"description":"{}","url":"{}","version":"{}","year":{},"contributor":"{}","date_created":"{}"}}'.format(
            self.description, self.url, self.version, self.year, self.contributor, self.date_created
        )

    def __repr__(self):
        return self.__str__()


class CocoLicense:
    def __init__(self, url, id, name):
        self.url = url
        self.id = id
        self.name = name

    def __str__(self):
        return '{{"url":"{}","id":{},"name":"{}"}}'.format(
            self.url, self.id, self.name
        )

    def __repr__(self):
        return self.__str__()


class CocoImage:
    def __init__(self, license, file_name, coco_url, height, width, date_captured, flickr_url, id):
        self.license = license
        self.file_name = file_name
        self.coco_url = coco_url
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.flickr_url = flickr_url
        self.id = id

    def __str__(self):
        return '{{"license":{},"file_name":"{}","coco_url":"{}","height":{},"width":{},"date_captured":"{}","flickr_url":"{}","id":{}}}'.format(
            self.license, self.file_name, self.coco_url, self.height, self.width, self.date_captured, self.flickr_url, self.id
        )

    def __repr__(self):
        return self.__str__()


class CocoAnnotation:
    def __init__(self, segmentation, area, iscrowd, image_id, bbox, category_id, id):
        self.segmentation = segmentation
        self.area = area
        self.iscrowd = iscrowd
        self.image_id = image_id
        self.bbox = bbox
        self.category_id = category_id
        self.id = id

    def __str__(self):
        if self.iscrowd == 0:
            return '{{"segmentation":{},"area":{},"iscrowd":{},"image_id":{},"bbox":{},"category_id":{},"id":{}}}'.format(
                self.segmentation, self.area, self.iscrowd, self.image_id, self.bbox, self.category_id, self.id
            )
        else:
            return '{{"segmentation":{{"counts":{},"size":{}}},"area":{},"iscrowd":{},"image_id":{},"bbox":{},"category_id":{},"id":{}}}'.format(
                self.segmentation['counts'], self.segmentation['size'], self.area, self.iscrowd, self.image_id, self.bbox, self.category_id, self.id
            )

    def __repr__(self):
        return self.__str__()


class CocoCategory:
    def __init__(self, supercategory, id, name):
        self.supercategory = supercategory
        self.id = id
        self.name = name

    def __str__(self):
        return '{{"supercategory":"{}","id":{},"name":"{}"}}'.format(
            self.supercategory, self.id, self.name
        )

    def __repr__(self):
        return self.__str__()

class Coco:
    def __init__(self, json_data):
        self.image_id_list = dict()
        self.next_id = 1

        # Coco info
        info_json = json_data['info']
        self.info = CocoInfo(info_json['description'], info_json['url'], info_json['version'],
                             info_json['year'], info_json['contributor'], info_json['date_created'])

        # License
        self.licenses = list()
        licenses_json = json_data['licenses']
        for license_json in licenses_json:
            coco_license = CocoLicense(license_json['url'], license_json['id'], license_json['name'])
            self.licenses.append(coco_license)

        # Images
        self.images = list()
        images_json = json_data['images']
        for image_json in images_json:
            coco_image = CocoImage(image_json['license'], image_json['file_name'], image_json['coco_url'],
                                   image_json['height'], image_json['width'], image_json['date_captured'],
                                   image_json['flickr_url'], image_json['id'])
            self.images.append(coco_image)
            self.image_id_list[coco_image.id] = True

        # Annotations
        self.annotations = list()
        self.annotation_id_list = dict()
        self.next_annotation_id = 1
        annotations_json = json_data['annotations']
        for annotation_json in annotations_json:
            coco_annotation = CocoAnnotation(annotation_json['segmentation'], annotation_json['area'],
                                             annotation_json['iscrowd'], annotation_json['image_id'],
                                             annotation_json['bbox'],
                                             annotation_json['category_id'], annotation_json['id'])
            self.annotations.append(coco_annotation)
            self.annotation_id_list[coco_annotation.id] = True

        # Category
        self.categories = list()
        self.max_category_id = 0
        categories_json = json_data['categories']
        for category_json in categories_json:
            coco_category = CocoCategory(category_json['supercategory'], category_json['id'], category_json['name'])
            self.max_category_id = max(self.max_category_id, coco_category.id)
            self.categories.append(coco_category)

    def get_new_image_id(self):
        while True:
            if self.next_id not in self.image_id_list.keys():
                self.image_id_list[self.next_id] = True
                self.next_id += 1
                return self.next_id - 1

            self.next_id += 1

    def get_category_id(self, category_name):
        for category in self.categories:
            if category.supercategory == category_name or category.name == category_name:
                return category.id
        return -1

    def add_category(self, supercategory, category_name):
        self.max_category_id += 1
        self.categories.append(CocoCategory(supercategory, self.max_category_id, category_name))
        return self.max_category_id

    def find_image_id_by_filename(self, filename):
        for image in self.images:
            if image.file_name == filename:
                return image.id
        return -1

    def add_image(self, license, file_name, coco_url, height, width, date_captured, flickr_url, id):
        self.images.append(CocoImage(license, file_name, coco_url, height, width, date_captured, flickr_url, id))
        return id

    def is_exist_image_id(self, id):
        for image in self.images:
            if image.id == id:
                return True
        return False

    def get_new_annotation_id(self):
        while True:
            if self.next_annotation_id not in self.annotation_id_list.keys():
                self.annotation_id_list[self.next_annotation_id] = True
                self.next_annotation_id += 1
                return self.next_annotation_id - 1

            self.next_annotation_id += 1

    def add_annotation(self, segmentation, area, iscrowd, image_id, bbox, category_id):
        id = self.get_new_annotation_id()
        self.annotations.append(CocoAnnotation(segmentation, area, iscrowd, image_id, bbox, category_id, id))
        return id

    def __str__(self):
        return '{{"info": {}, "licenses": {}, "images": {}, "annotations": {}, "categories": {}}}'.format(
            self.info, self.licenses, self.images, self.annotations, self.categories
        )

    def __repr__(self):
        return self.__str__()




