import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desabilita GPU ANTES de importar TF/DeepFace

from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import uuid # Para nomes de arquivo unicos
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Libera CORS para todas as origens

# Configura logging
app.logger.setLevel(logging.INFO) # Use INFO ou DEBUG
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)


# Pasta para armazenar as imagens temporárias
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    app.logger.info("Rota / acessada")
    return 'API de Reconhecimento Facial AB2L no ar!'

@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    app.logger.info("========= ROTA /reconhecer INICIADA =========") # <--- ESTA É A NOVA LINHA DE LOG ADICIONADA
    app.logger.info("Conteúdo de request.files: %s", request.files) # <--- ADICIONE ESTE LOG TAMBÉM para ver se o arquivo chega
    app.logger.info("Conteúdo de request.form: %s", request.form)   # <--- E ESTE para ver outros dados do formulário

    if 'imagem' not in request.files:
        app.logger.warning("Nenhuma imagem enviada no request.files")
        return jsonify({"erro": "Envie uma imagem"}), 400

    ref_file = request.files['imagem']
    app.logger.info(f"Arquivo de imagem recebido: {ref_file.filename}, mimetype: {ref_file.mimetype}")
    
    # Gera um nome de arquivo único para a imagem de referência
    file_extension = os.path.splitext(ref_file.filename)[1]
    unique_ref_filename = str(uuid.uuid4()) + file_extension
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_ref_filename)
    
    temp_files_to_clean = [ref_path] # Lista de arquivos para limpar no final

    try:
        ref_file.save(ref_path)
        app.logger.info(f"Imagem de referencia salva em: {ref_path}")

        # Chama a API do seu site WordPress
        wp_api_url = 'https://dev-ab2l.pantheonsite.io/wp-json/meus/v1/eventos'
        app.logger.info(f"Buscando eventos de: {wp_api_url}")
        
       eventos_response = requests.get(wp_api_url, timeout=15) # Timeout de 15s
eventos_response.raise_for_status() # Levanta erro para status 4xx/5xx
eventos_originais = eventos_response.json() # Renomeie para não confundir

# PARA TESTE: Processe apenas o primeiro evento
eventos = eventos_originais[:1] 
app.logger.info(f"Recebidos {len(eventos_originais)} eventos do WordPress, processando APENAS {len(eventos)} para teste.")
        
        if not isinstance(eventos, list):
            app.logger.error(f"API do WordPress nao retornou uma lista. Resposta: {eventos}")
            return jsonify({"erro": "Formato de resposta inesperado da API do WordPress."}), 500
        
        app.logger.info(f"Recebidos {len(eventos)} eventos do WordPress.")
        encontrados = []

        for evento in eventos:
            foto_url = evento.get('foto_url')
            if not foto_url:
                app.logger.warning(f"Evento {evento.get('id', 'ID Desconhecido')} sem foto_url. Pulando.")
                continue

            # Gera nome unico para imagem do evento baixada
            event_img_extension = os.path.splitext(foto_url.split('/')[-1])[1]
            if not event_img_extension: # Se a URL não tiver extensão, assume .jpg
                event_img_extension = ".jpg"
            unique_event_img_filename = str(uuid.uuid4()) + event_img_extension
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_event_img_filename)
            temp_files_to_clean.append(img_path)

            try:
                app.logger.info(f"Baixando imagem do evento: {foto_url}")
                img_data_response = requests.get(foto_url, timeout=15)
                img_data_response.raise_for_status()
                
                with open(img_path, 'wb') as f:
                    f.write(img_data_response.content)
                app.logger.info(f"Imagem do evento salva em: {img_path}")

                app.logger.info(f"Comparando {ref_path} com {img_path}")
                result = DeepFace.verify(
                    img1_path=ref_path,
                    img2_path=img_path,
                    model_name="VGG-Face", # Pode especificar modelos: "VGG-Face", "Facenet", "ArcFace" etc.
                    enforce_detection=False # Mantenha False se as imagens já estiverem bem cortadas no rosto
                )
                app.logger.info(f"Resultado da verificacao para {foto_url}: {result}")

                if result.get('verified'): # Usar .get() para evitar KeyError
                    encontrados.append(evento)
                    app.logger.info(f"Rosto encontrado no evento: {evento.get('title', evento.get('id', 'Detalhes do Evento'))}")

            except requests.exceptions.RequestException as e_req:
                app.logger.error(f"Erro de rede ao baixar/processar {foto_url}: {e_req}")
                continue # Pula para o proximo evento
            except Exception as e_df:
                app.logger.error(f"Erro no DeepFace ao comparar com {foto_url}: {e_df}")
                # Considerar se deve continuar ou parar; por enquanto, continua
                continue
        
        app.logger.info(f"Processamento concluido. Encontradas {len(encontrados)} correspondencias.")
        return jsonify({"encontrado_em": encontrados})

    except requests.exceptions.RequestException as e_wp:
        app.logger.error(f"Erro ao buscar eventos do WordPress ({wp_api_url}): {e_wp}")
        return jsonify({"erro": f"Nao foi possivel buscar eventos do WordPress: {str(e_wp)}"}), 503 # Service Unavailable
    except ValueError as e_json: # Erro ao decodificar JSON
        app.logger.error(f"Erro ao decodificar JSON da API do WordPress: {e_json}")
        return jsonify({"erro": "Resposta invalida (nao JSON) da API do WordPress."}), 500
    except Exception as e_geral:
        app.logger.error(f"Erro inesperado na rota /reconhecer: {e_geral}", exc_info=True)
        return jsonify({"erro": f"Ocorreu um erro interno no servidor: {str(e_geral)}"}), 500
    finally:
        # Limpar arquivos temporários
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    app.logger.info(f"Arquivo temporario removido: {f_path}")
                except Exception as e_clean:
                    app.logger.error(f"Erro ao remover arquivo temporario {f_path}: {e_clean}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Para debug local, pode usar app.run(host="0.0.0.0", port=port, debug=True)
    # No Render, o Gunicorn vai chamar 'app:app', então esta parte é mais para local.
    # O Gunicorn não usa o app.run()
    app.run(host="0.0.0.0", port=port) # Gunicorn vai ignorar isso, mas é bom pra teste local
