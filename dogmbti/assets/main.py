'''
#pd로 저장
from tensorflow import keras
model = keras.models.load_model('C:/A_Projects/dogmbti_app/dogmbti/assets/dog_not_dog_v3.h5', compile=False)
 
export_path = 'C:/A_Projects/dogmbti_app/dogmbti/assets/dog_not_dog'
model.save(export_path, save_format="tf")
'''


#tflite로 저장
import tensorflow as tf
 
saved_model_dir = 'C:/A_Projects/dogmbti_app/dogmbti/assets/dog_not_dog'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('C:/A_Projects/dogmbti_app/dogmbti/assets/dog_not_dog_v3.tflite', 'wb').write(tflite_model)