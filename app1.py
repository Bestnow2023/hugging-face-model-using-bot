from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

model_name = 'jarradh/llama2_70b_chat_uncensored'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def chatbot(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


gr.Interface(
    fn=chatbot,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox()
).launch()
