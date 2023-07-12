from keras.models import load_model
import numpy as np
import pickle

model = load_model('classifier_model.h5')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

with open("generated_image.pickle", "rb") as file:
    img = pickle.load(file)

result = model.predict(img)

max_high_index = np.argmax(result)

predicted_digit = max_high_index

print(f"Predicted digit: {predicted_digit}")
