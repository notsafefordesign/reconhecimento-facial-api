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
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# --- MODIFICAÇÃO PARA CAMINHOS ABSOLUTOS ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER_NAME = "uploads" # Nome da pasta
UPLOAD_FOLDER_ABSOLUTE = os.path.join(BASE_DIR, UPLOAD_FOLDER_NAME)
os.makedirs(UPLOAD_FOLDER_ABSOLUTE, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_ABSOLUTE # Usar caminho absoluto
# --- FIM DA MODIFICAÇÃO ---

@app.route('/')
def home():
    app.logger.info("Rota / acessada")
    return 'API de Reconhecimento Facial AB2L no ar!'

@app.route('/reconhecer', methods=['POST'])
def reconhecer():
    app.logger.info("========= ROTA /reconhecer INICIADA =========")
    app.logger.info("Conteúdo de request.files: %s", request.files)
    app.logger.info("Conteúdo de request.form: %s", request.form)

    if 'imagem' not in request.files:
        app.logger.warning("Nenhuma imagem enviada no request.files")
        return jsonify({"erro": "Envie uma imagem"}), 400

    ref_file = request.files['imagem']
    app.logger.info(f"Arquivo de imagem recebido: {ref_file.filename}, mimetype: {ref_file.mimetype}")
    
    file_extension = os.path.splitext(ref_file.filename)[1]
    unique_ref_filename = str(uuid.uuid4()) + file_extension
    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_ref_filename)
    
    temp_files_to_clean = [ref_path]

    try:
        ref_file.save(ref_path)
        app.logger.info(f"Imagem de referencia salva em (abs): {ref_path}")

        # --- VERIFICAÇÃO DE EXISTÊNCIA PARA REF_PATH ---
        if not os.path.exists(ref_path):
            app.logger.error(f"ERRO CRÍTICO: Arquivo de referência {ref_path} NÃO existe no disco após salvar. Abortando.")
            return jsonify({"erro": "Erro interno ao salvar imagem de referência."}), 500
        # --- FIM DA VERIFICAÇÃO ---

        wp_api_url = 'https://dev-ab2l.pantheonsite.io/wp-json/meus/v1/eventos'
        app.logger.info(f"Buscando eventos de: {wp_api_url}")
        
        eventos_response = requests.get(wp_api_url, timeout=15)
        eventos_response.raise_for_status()
        eventos_originais = eventos_response.json()
        
        eventos = eventos_originais[:1] # PARA TESTE: Processe apenas o primeiro evento
        app.logger.info(f"Recebidos {len(eventos_originais)} eventos do WordPress, processando APENAS {len(eventos)} para teste.")
        
        if not isinstance(eventos_originais, list):
            app.logger.error(f"API do WordPress nao retornou uma lista. Resposta: {eventos_originais}")
            return jsonify({"erro": "Formato de resposta inesperado da API do WordPress."}), 500
        
        encontrados = []

        for evento in eventos:
            foto_url = evento.get('foto_url')
            if not foto_url:
                app.logger.warning(f"Evento {evento.get('id', 'ID Desconhecido')} sem foto_url. Pulando.")
                continue

            event_img_extension = os.path.splitext(foto_url.split('/')[-1])[1]
            if not event_img_extension:
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
                app.logger.info(f"Imagem do evento salva em (abs): {img_path}")

                # --- VERIFICAÇÃO DE EXISTÊNCIA PARA IMG_PATH ---
                if not os.path.exists(img_path):
                    app.logger.error(f"ERRO CRÍTICO: Arquivo do evento {img_path} NÃO existe no disco após salvar. Pulando imagem {foto_url}.")
                    temp_files_to_clean.remove(img_path) # Não foi usado, não precisa limpar depois
                    continue 
                # --- FIM DA VERIFICAÇÃO ---

                app.logger.info(f"Comparando (abs) {ref_path} com (abs) {img_path} usando SFace")
                result = DeepFace.verify(
                    img1_path=ref_path,
                    img2_path=img_path,
                    model_name="SFace",  # Usando SFace
                    enforce_detection=False
                )
                app.logger.info(f"Resultado da verificacao para {foto_url} (SFace): {result}")

                if result.get('verified'):
                    encontrados.append(evento)
                    app.logger.info(f"Rosto encontrado no evento (SFace): {evento.get('title', evento.get('id', 'Detalhes do Evento'))}")

            except requests.exceptions.RequestException as e_req:
                app.logger.error(f"Erro de rede ao baixar/processar {foto_url}: {e_req}")
                continue
            except Exception as e_df: # Captura erros do DeepFace ou outros inesperados aqui
                app.logger.error(
                    f"Erro no DeepFace (SFace) ou outro ao comparar com {foto_url}. Detalhe do erro: {str(e_df)}",
                    exc_info=True  # Loga o traceback completo
                )
                continue
        
        app.logger.info(f"Processamento concluido. Encontradas {len(encontrados)} correspondencias.")
        return jsonify({"encontrado_em": encontrados})

    except requests.exceptions.RequestException as e_wp:
        app.logger.error(f"Erro ao buscar eventos do WordPress ({wp_api_url}): {e_wp}")
        return jsonify({"erro": f"Nao foi possivel buscar eventos do WordPress: {str(e_wp)}"}), 503
    except ValueError as e_json:
        app.logger.error(f"Erro ao decodificar JSON da API do WordPress: {e_json}")
        return jsonify({"erro": "Resposta invalida (nao JSON) da API do WordPress."}), 500
    except Exception as e_geral:
        app.logger.error(f"Erro inesperado na rota /reconhecer: {e_geral}", exc_info=True)
        return jsonify({"erro": f"Ocorreu um erro interno no servidor: {str(e_geral)}"}), 500
    finally:
        app.logger.info(f"Iniciando limpeza de {len(temp_files_to_clean)} arquivos temporários.")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    app.logger.info(f"Arquivo temporario removido: {f_path}")
                except Exception as e_clean:
                    app.logger.error(f"Erro ao remover arquivo temporario {f_path}: {e_clean}")
            else:
                app.logger.warning(f"Tentativa de remover arquivo temporario inexistente: {f_path}")


if __name__ == "__main__":
    # Esta parte é principalmente para debug local, Gunicorn não a usa diretamente no Render.
    # O Render usa o 'Start Command' do dashboard (ou render.yaml) que chama 'gunicorn app:app ...'
    port = int(os.environ.get("PORT", 10000)) # Render define a variável PORT
    # Para debug local: app.run(host="0.0.0.0", port=port, debug=True)
    # Para produção com Gunicorn, o Gunicorn lida com o host e porta.
    # Se rodar 'python app.py' localmente, isso funcionará.
    app.run(host="0.0.0.0", port=port)
