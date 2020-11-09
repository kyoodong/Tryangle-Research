

CLASSES = ("alleys", "bridge", "building_front", "handrail", "indoor", "indoor_window",
           "mountain", "plain", "sea", "sea_sunset", "traditional_house", "treeroad",
           "treeroad_autumn", "treeroad_spring", "tunnel", "wall")

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)



# ----------------------- DATASETS ----------------------- #

dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': './data/images/',

    # Validation images and annotations.
    'valid_images': './data/images/',

    # A list of names for each of you classes.
    'class_names': CLASSES,

})

# ------------------- BACKBONE CONFIGS ------------------ #

efficientnet_config = Config({
    "model_name": "efficientnet-b4",
    "include_top": False
})

# -------------------- MODEL CONFIGS -------------------- #

model_base_config = Config({
    'name': 'model_base',
    
    "backbone": efficientnet_config.copy(),

    "dataset": dataset_base,
    "num_classes": len(dataset_base.class_names) + 1,
    "max_size": 380,

    "dropout_rate": 0.4,

    # Training params
    'lr': 1e-3,
    'momentum': 0.9,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,

    'max_iter': 8000,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, this will resize all images to be max_size^2 pixels in area while keeping aspect ratio.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,
})



# Default config
cfg = model_base_config.copy()
