from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import pickle

# Generator modelını yükle
generator = load_model('generator_model.h5')

# Gürültü vektörü oluştur
noise = np.random.normal(0, 1, (1, 100))

# Görüntüyü üret
generated_image = generator.predict(noise)

# Üretilen görüntüyü kaydet

with open("generated_image.pickle","wb") as file:
    pickle.dump(generated_image, file)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.savefig('generated_image.png', bbox_inches='tight')

