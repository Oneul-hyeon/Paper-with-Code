import gradio as gr
from translator import Translator

def translate(sequence) :
    return translator.translate(sequence)

if __name__ == "__main__" :
    translator = Translator()
    with gr.Blocks() as demo :
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div><h1>Transformer 번역 모델 테스트 Demo</h1></div>
            <div>by Oneul-hyeon</div>
            </div>""")
        with gr.Row() :
            with gr.Column() :
                input = gr.Textbox(label = "한국어를 입력해주세요.", interactive=True)
                translation_button = gr.Button("번역하기")
            with gr.Column() :
                output = gr.Textbox(label = "번역 결과", interactive=False)
            translation_button.click(fn=translate, inputs=input, outputs=output)
    
    demo.launch(debug = True, server_name = "0.0.0.0", server_port = 12399)