import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Carregar e preparar os dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Construir o modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo e medir o tempo de treinamento
start_time = time.time()
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
training_time = time.time() - start_time

# Salvar o modelo
model.save('model/pesos.h5')

# Avaliar o modelo
_, accuracy = model.evaluate(x_test, y_test)

# Medir o tempo de inferência
start_time = time.time()
model.predict(x_test[:1])
inference_time = time.time() - start_time

print(f"Acurácia do modelo CNN: {accuracy * 100:.2f}%")
print(f"Tempo de Treinamento do modelo CNN: {training_time:.2f} segundos")
print(f"Tempo de Inferência do modelo CNN: {inference_time:.6f} segundos por imagem")
