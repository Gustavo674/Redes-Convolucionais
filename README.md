# Algoritmo de Reconhecimento de Dígitos

Este projeto utiliza redes neurais convolucionais (CNN) e um modelo linear para reconhecer dígitos manuscritos. O objetivo é comparar o desempenho de ambos os modelos em termos de acurácia, tempo de treinamento e tempo de inferência.

## Estrutura do Projeto

```plaintext
Redes-Convolucionais/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── static/
│   │   └── css/
│   │       └── style.css
│   └── templates/
│       └── upload.html
│
├── model/
│   ├── train_model.py
│   ├── train_model_linear.py
│   ├── pesos.h5
│   └── model_linear.pkl
│
├── requirements.txt
├── run.py
└── README.md
```

## Instalação
1. Clone o repositório:

```sh
git clone https://github.com/usuario/Redes-Convolucionais.git
cd Redes-Convolucionais
```

2. Instale as dependências:
```sh
pip install -r requirements.txt
```

### Treine os modelos

1. Modelo CNN:

```sh
python3 model/train_model.py
```

2. Modelo Linear:

```sh
python3 model/train_model_linear.py
```

3. Execute a aplicação:

```sh
python3 run.py
```

## Rotas

- /predict (POST): Recebe uma imagem e retorna a predição dos dígitos pelos modelos CNN e Linear.

    - Entrada: Imagem de um dígito manuscrito.
    - Saída: JSON com as predições dos modelos CNN e Linear.

```json
{
  "digit_cnn": 5,
  "digit_linear": 5
}
```

- /upload (GET): Exibe uma página HTML para upload de imagem.

    - Descrição: Página com um formulário para envio de imagem.

## Comparação dos Modelos

### Modelo Convolucional (CNN)
- Acurácia: 98.61%
- Tempo de Treinamento: 157.60 segundos
- Tempo de Inferência: 0.107223 segundos por imagem

### Modelo Linear (Regressão Logística)
- Acurácia: 92.62%
- Tempo de Treinamento: 39.82 segundos
- Tempo de Inferência: 0.000275 segundos por imagem

## Vídeo de Demonstração

[Link para o vídeo](https://drive.google.com/file/d/1P-lHtkB-Yk-gB5lz7HMZlsTbJm1IfDme/view?usp=sharing)
