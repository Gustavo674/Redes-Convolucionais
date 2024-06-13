import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
import joblib

# Carregar e preparar os dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Treinar o modelo de regressão logística e medir o tempo de treinamento
start_time = time.time()
model_linear = LogisticRegression(max_iter=1000)
model_linear.fit(x_train, y_train)
training_time = time.time() - start_time

# Salvar o modelo
joblib.dump(model_linear, 'model/model_linear.pkl')

# Avaliar o modelo
accuracy = model_linear.score(x_test, y_test)

# Medir o tempo de inferência
start_time = time.time()
model_linear.predict(x_test[:1])
inference_time = time.time() - start_time

print(f"Acurácia do modelo Linear: {accuracy * 100:.2f}%")
print(f"Tempo de Treinamento do modelo Linear: {training_time:.2f} segundos")
print(f"Tempo de Inferência do modelo Linear: {inference_time:.6f} segundos por imagem")
