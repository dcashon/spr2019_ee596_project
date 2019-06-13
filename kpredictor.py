from keras.models import Model, load_model




class Predictor():
    def __init__(self):
        """
        Loads the trained regression and classification models and contructs the appropriate models for 
        arbitrary image input
        """
        r = load_model('/mnt/disks/gscratch/regressor_redo2_epoch4.h5')
        c = load_model('/mnt/disks/gscratch/my_model2_epoch1.h5')
        self.regressor = Model(inputs=regressor.layers[0].input, outputs=regressor.layers[-2].output)
        self.classifier = Model(inputs=classifier.layers[0].input, outputs=classifier.layers[-3].output)
    def predict_with_details():
        """
        Shows layer shapes for the prediction
        Helpful to see whats happening
        """
        

    def multi_scale_prediction(input_img, num_scales):
        """
        Attempts object localization on input image using num_scales
        """

