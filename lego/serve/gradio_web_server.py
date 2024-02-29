import shutil
import torch
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
import uvicorn
from lego.conversation import conv_templates,  Conversation
from lego.serve.gradio_utils import Chat, tos_markdown, learn_more_markdown, title_markdown, block_css, colors
from lego.model.builder import CONFIG
from lego.mm_utils import load_image_square, postprocess_output, postprocess_image_answer_gradio
from video_llama.processors.video_processor import load_video
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from lego.constants import  DEFAULT_IMAGE_PATCH_TOKEN,DEFAULT_IMAGE_START_TOKEN,DEFAULT_IMAGE_END_TOKEN,DEFAULT_VIDEO_PATCH_TOKEN,\
                            DEFAULT_VIDEO_START_TOKEN,DEFAULT_VIDEO_END_TOKEN,DEFAULT_SOUND_PATCH_TOKEN,DEFAULT_SOUND_START_TOKEN,DEFAULT_SOUND_END_TOKEN

def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    return filename

def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename

def save_sound_to_local(sound_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.flac')
    shutil.copyfile(sound_path, filename)
    return filename


def generate(image1, video, sound, textbox_in, first_run, state, state_, images_tensor):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    image1 = image1 if image1 else "none"
    video = video if video else "none"
    sound = sound if sound else "none"


    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = [[], []]

    first_run = False if len(state.messages) > 0 else True

    text_en_in = textbox_in.replace("picture", "image")

    # images_tensor = [[], []]
    image_processor = handler.image_processor
    video_transform = handler.video_transform
    if first_run:
        if os.path.exists(image1):
            loaded_image = load_image_square(image1,image_processor)
            tensor = image_processor.preprocess(loaded_image, return_tensors='pt')['pixel_values'].half().cuda()
            # print(tensor.shape)
            tensor = tensor.to(handler.model.device, dtype=dtype)
            images_tensor[0] = images_tensor[0] + [tensor]
            images_tensor[1] = images_tensor[1] + ['image']
        if os.path.exists(video):
            loaded_video = load_video(
                    video_path = video,
                    n_frms = 64,
                    height = 224,
                    width = 224,
                    sampling ="uniform", return_msg = False
                )
            tensor = video_transform(loaded_video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
            # print(tensor.shape)
            tensor = tensor.to(handler.model.device, dtype=dtype)
            images_tensor[0] = images_tensor[0] + [tensor]
            images_tensor[1] = images_tensor[1] + ['video']
        if os.path.exists(sound):
            tensor = load_and_transform_audio_data([sound],device="cpu")
            tensor = tensor.to(handler.model.device, dtype=dtype)
            images_tensor[0] = images_tensor[0] + [tensor]
            images_tensor[1] = images_tensor[1] + ['sound']

        if os.path.exists(image1):
            text_en_in = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len + DEFAULT_IMAGE_END_TOKEN  + '\n' + text_en_in
        if os.path.exists(video):
            text_en_in = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len + DEFAULT_VIDEO_END_TOKEN + '\n' + text_en_in
        if os.path.exists(sound):
            text_en_in = DEFAULT_SOUND_START_TOKEN + DEFAULT_SOUND_PATCH_TOKEN * CONFIG.sound_token_len + DEFAULT_SOUND_END_TOKEN + '\n' + text_en_in
    text_en_out, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_)
    state_.messages[-1][-1] = text_en_out# state_.messages[-1] = (state_.roles[1], text_en_out)
    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out
    result_path = None
    if image1 is not None and '[' in textbox_out:
        textbox_out = postprocess_output(textbox_out,image1)
        textbox_out,result_path = postprocess_image_answer_gradio(textbox_out, image1)

    show_images = ""
    if os.path.exists(image1):
        filename = save_image_to_local(image1)
        show_images += f'<img src="./file={filename}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if os.path.exists(video):
        filename = save_video_to_local(video)
        show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'
    if os.path.exists(sound):
        filename = save_sound_to_local(sound)
        show_images += f'<audio controls src="./file={filename}"></audio>'
    if flag:
        state.append_message(state.roles[0], textbox_in + "\n" + show_images)
    if result_path:
        result_images =f'<img src="./file={result_path}" style="display: inline-block;width: 250px;max-height: 400px;">'
    state.append_message(state.roles[1], textbox_out+ "\n" + result_images if result_path else textbox_out)
    # return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True))
    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=image1 if os.path.exists(image1) else None, interactive=True), gr.update(value=video if os.path.exists(video) else None, interactive=True), gr.update(value=sound if os.path.exists(sound) else None, interactive=True))

def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        True, state, state_, state.to_gradio_chatbot(), [[], []])


conv_mode = "default"
model_path = '/ckpt/LEGO-7B'
device = 'cuda'
load_8bit = False
load_4bit = False
dtype = torch.bfloat16
handler = Chat(model_path, conv_mode=conv_mode, load_8bit=load_8bit, load_4bit=load_8bit, device=device)
if not os.path.exists("temp"):
    os.makedirs("temp")

app = FastAPI()

textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
with gr.Blocks(title='GroundingGPT🚀', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            image1 = gr.Image(label="Input Image", type="filepath")
            video = gr.Video(label="Input Video", type="filepath")
            sound = gr.Audio(label="Input Audio", type="filepath")
            # cur_dir = os.path.dirname(os.path.abspath(__file__))
            cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/lake.jpeg",
                        "What are the things I should be cautious about when I visit here?",
                    ],
                    [
                        f"{cur_dir}/examples/pizza.jpg",
                        "Please describe the image and include the object positions in [x0, y0, x1, y1] format.",
                    ],
                    [
                        f"{cur_dir}/examples/COCO_train2014_000000580668.jpg",
                        "Describe the image and provide the coordinates for the objects mentioned.",
                    ],
                ],
                inputs=[image1, textbox],
            )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="GroundingGPT", bubble_full_width=True).style(height=900)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn = gr.Button(value="👎  Downvote", interactive=True)
                flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    f"{cur_dir}/examples/girlreading.mp4",
                    "Describe this video.",
                ],
                [
                    f"{cur_dir}/examples/baby.mp4",
                    "Why is this video funny?",
                ],
                [
                    f"{cur_dir}/examples/city.mp4",
                    "Which city is this do you think? Introduce it to me.",
                ],
            ],
            inputs=[video, textbox],
        )
        with gr.Column():
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/Machine.wav",
                        "What can you hear in this audio?",
                    ],
                    [
                        f"{cur_dir}/examples/Rain.wav",
                        "Describe this audio please."
                    ],
                ],
                inputs=[sound, textbox],
            )
            gr.Examples(
                examples=[
                    [   
                        f"{cur_dir}/examples/dog.jpeg",
                        f"{cur_dir}/examples/dog.flac",
                        "Where is the sound's point of emission in this image? Include the object positions in [x0, y0, x1, y1] format.",
                    ] 
                ],
                inputs=[image1, sound, textbox],
            )
    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)

    submit_btn.click(generate, [image1, video, sound, textbox, first_run, state, state_, images_tensor],
                     [state, state_, chatbot, first_run, textbox, images_tensor, image1, video, sound])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        generate, [image1, video, sound, textbox, first_run, state, state_, images_tensor], [state, state_, chatbot, first_run, textbox, images_tensor, image1, video, sound])

    clear_btn.click(clear_history, [state, state_],
                    [image1, video, sound, textbox, first_run, state, state_, chatbot, images_tensor])
demo.launch(share=True)
