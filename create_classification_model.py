from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# MNIST veri setini yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Görüntüleri normalize et ve boyutunu düzenle
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Etiketleri kategorik hale getir
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modeli oluştur
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation =   'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'sigmoid'))

# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# Modeli derle
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Modeli değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Modeli kaydet
model.save('classifier_model.h5')
