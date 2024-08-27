import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Definisikan ukuran gambar
img_width, img_height = 150, 150

# Memuat model
model = load_model('model/inceptionv3_nsfw_model.h5')

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_names = ['sexy', 'hentai', 'porn', 'neutral', 'drawings']
    result = dict(zip(class_names, predictions[0].astype(float)))

    return json.dumps(result, indent=2)

# Contoh penggunaan
print(classify_image("img/hehe.jpg"))