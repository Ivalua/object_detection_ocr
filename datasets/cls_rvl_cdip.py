from keras.preprocessing.image import ImageDataGenerator

class Dataset:

    def __init__(self, batch_size=3, input_dim=150, **kwargs):
        local_keys = locals()
        self.enable_classification = True
        self.enable_boundingbox = False
        self.enable_segmentation = False
        
        #classes
        self.classes = ["letter","form", "email", "handwritten", "advertisement", \
            "scientific_report", "scientific_publication", "specification", \
            "file_folder", "news_article", "budget", "invoice", \
            "presentation", "questionnaire", "resume", "memo" ]
        self.num_classes = len(self.classes)
        print("Nb classes: " + str(self.num_classes))

        self.img_h = input_dim
        self.img_w = input_dim
        self.input_shape = ( self.img_h, self.img_w , 3)
        # self.stride_margin = True
        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train = train_datagen.flow_from_directory('/sharedfiles/rvl_cdip/train',
                                                     target_size=(self.img_w, self.img_h),
                                                     batch_size=batch_size,
                                                     class_mode='categorical', classes=self.classes)

        self.val = test_datagen.flow_from_directory('/sharedfiles/rvl_cdip/val',
                                                     target_size=(self.img_w, self.img_h),
                                                batch_size=batch_size,
                                                class_mode='categorical', classes=self.classes)

        self.test = test_datagen.flow_from_directory('/sharedfiles/rvl_cdip/test',
                                                     target_size=(self.img_w, self.img_h),
                                                batch_size=batch_size,
                                                class_mode='categorical', classes=self.classes)
        # for compatibility
        self.gt_test = []
        self.stride_margin = 0
