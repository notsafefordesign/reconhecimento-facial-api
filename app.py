import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desabilita o uso da GPU


from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Libera CORS para todas as origens

# Pasta para armazenar as imagens temporárias
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Definir a pasta de uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return 'API no ar!'

@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    # Verifica se o arquivo foi enviado
    if 'imagem' not in request.files:
        return jsonify({"erro": "Envie uma imagem"}), 400

    ref = request.files['imagem']
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref.filename)
    ref.save(ref_path)

    # Chama a API do seu site WordPress que retorna os eventos com fotos
    eventos = requests.get('https://dev-ab2l.pantheonsite.io/wp-json/meus/v1/eventos').json()
    encontrados = []

    for evento in eventos:
        url = evento.get('foto_url')
        if not url:
            continue

        nome_img = url.split('/')[-1]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], nome_img)

        try:
            # Baixa a imagem do evento
            img_data = requests.get(url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)

            # Realiza a comparação facial
            result = DeepFace.verify(
                img1_path=ref_path,
                img2_path=img_path,
                enforce_detection=False
            )

            # Se as imagens forem iguais, adiciona o evento à lista
            if result['verified']:
                encontrados.append(evento)

        except Exception as e:
            print(f"Erro ao comparar com {url}: {e}")
            continue

    return jsonify({"encontrado_em": encontrados})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

