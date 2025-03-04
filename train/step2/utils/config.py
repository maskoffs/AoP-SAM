class Config_PSFH:
    # This dataset is for breast cancer segmentation
    data_path = "./dataset/"  #
    save_path = "./checkpoints/"
    result_path = "./result/"
    tensorboard_path = "./tensorboard/"
    load_path = "./checkpoints/"
    save_path_code = "_"

    workers = 0
    epochs = 30 # number of total epochs to run (default: 400)
    batch_size = 8  # batch size (default: 4)
    learning_rate = 1e-4  # iniial learning rate (default: 0.001)
    momentum = 0.9  # momntum
    classes = 3  # thenumber of classes (background + foreground)
    img_size = 256  # theinput size of model
    train_split = "train_new"  # the file name of training set
    val_split = "val_new"  # the file name of testing set
    test_split = "test"  # the file name of testing set # HMCQU
    crop = None  # the cropped image size
    eval_freq = 2  # the frequency of evaluate the model
    save_freq = 10  # the frequency of saving the model
    device = "cuda"  # training device, cpu or cuda
    cuda = "on"  # switch on/off cuda option (default: off)
    gray = "no"  # the type of input image
    img_channel = 3
    eval_mode = "slice"
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "AoP-SAM"
    device = "cuda"

# ==================================================================================================
def get_config(task="PSFH"):
    if task == "PSFH":
        return Config_PSFH()
    else:
        assert("We do not have the related dataset, please choose another task.")