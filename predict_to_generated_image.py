from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pickle

model = load_model('classifier_model.h5')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


with open("generated_image.pickle", "rb") as file:
    img = pickle.load(file)

#predict the result
result = model.predict(img)

# print("Üretilen görüntü rakamı:", result)

en_yuksek_indeks = np.argmax(result)

# Tahmin edilen rakam
tahmin_edilen_rakam = en_yuksek_indeks

print("Tahmin edilen rakam:", tahmin_edilen_rakam)
