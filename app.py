import gradio as gr
import os
from rag import procesar_pdf

# Variable global para la cadena QA activa
cadena_activa    = None
nombre_documento = None

def cargar_pdf(archivo):
    global cadena_activa, nombre_documento

    if archivo is None:
        return 'No se subio ningun archivo.', gr.update(interactive=False), gr.update(interactive=False)

    try:
        cadena_activa, total_fragmentos = procesar_pdf(archivo.name)
        nombre_documento = os.path.basename(archivo.name)
        mensaje = f'Documento cargado: {nombre_documento}\nFragmentos procesados: {total_fragmentos}\nYa puedes hacer preguntas.'
        return mensaje, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        return f'Error al cargar el documento: {str(e)}', gr.update(interactive=False), gr.update(interactive=False)

def responder_pregunta(pregunta, historial):
    global cadena_activa

    if historial is None:
        historial = []

    if cadena_activa is None:
        historial = historial + [
            {'role': 'user', 'content': pregunta},
            {'role': 'assistant', 'content': 'Primero debes cargar un documento PDF.'}
        ]
        return historial

    if not pregunta.strip():
        return historial

    try:
        respuesta = cadena_activa.invoke(pregunta)
    except Exception as e:
        respuesta = f'Error al procesar la pregunta: {str(e)}'

    historial = historial + [
        {'role': 'user', 'content': pregunta},
        {'role': 'assistant', 'content': respuesta}
    ]
    return historial

def limpiar_chat():
    return []

# Interfaz Gradio
with gr.Blocks(title='Chatbot RAG') as interfaz:
    gr.Markdown('# Chatbot RAG — Pregunta sobre tus documentos')
    gr.Markdown('Sube un PDF y haz preguntas sobre su contenido')

    with gr.Row():
        with gr.Column(scale=1):
            entrada_pdf   = gr.File(label='Subir PDF', file_types=['.pdf'])
            boton_cargar  = gr.Button('Cargar documento', variant='primary')
            estado_carga  = gr.Textbox(label='Estado', lines=3, interactive=False)
            boton_limpiar = gr.Button('Limpiar chat')

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label='Conversacion',
                height=400,
                type='messages'
            )
            entrada_pregunta = gr.Textbox(
                label='Pregunta',
                placeholder='Escribe tu pregunta aqui...',
                interactive=False
            )
            boton_preguntar = gr.Button('Enviar', variant='primary', interactive=False)

    boton_cargar.click(
        fn=cargar_pdf,
        inputs=entrada_pdf,
        outputs=[estado_carga, entrada_pregunta, boton_preguntar]
    )

    boton_preguntar.click(
        fn=responder_pregunta,
        inputs=[entrada_pregunta, chatbot],
        outputs=chatbot
    )

    entrada_pregunta.submit(
        fn=responder_pregunta,
        inputs=[entrada_pregunta, chatbot],
        outputs=chatbot
    )

    boton_limpiar.click(
        fn=limpiar_chat,
        outputs=chatbot
    )

interfaz.launch()