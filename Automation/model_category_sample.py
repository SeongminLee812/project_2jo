import tensorflow_hub as hub
import tensorflow as tf

class ModelCategory():
    def __init__(self):
        self.model_dict = {}

    def set_model(self):
        self.model_dict['wallet'] = {'model_name':'R50x1_object', 'model':self.__get_model_build('https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1')}
        self.model_dict['phone'] = {'model_name':'Efficientnet_b0', 'model':self.__get_model_build('https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2')}

    def __get_model_build(self, model_url):
        layer = hub.KerasLayer(model_url, input_shape=(224, 224) + (3,))
        model = tf.keras.Sequential([layer])
        model.build([None, 244, 244, 3])

        return model