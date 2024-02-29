import argparse
import torch
from lego.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_SOUND_TOKEN
from lego.conversation import SeparatorStyle
from lego import conversation as conversation_lib
from lego.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, load_image_square, postprocess_output
from lego.model.builder import CONFIG, load_pretrained_model
from video_llama.processors.video_processor import load_video
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from lego.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
                           DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN, DEFAULT_SOUND_PATCH_TOKEN, DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN

def main(args):
    model, tokenizer, image_processor, video_transform, context_len = load_pretrained_model(args.model_path)
    conv = conversation_lib.default_conversation.copy()
    roles = conv.roles
    image_path = None
    image_tensor = None
    video_tensor = None
    sound_tensor = None
    while True:
        image = None
        video = None
        sound = None
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if inp == 'change image':
            while True:
                try :
                    image_path = input('Please input new image path:')
                    image = load_image_square(image_path,image_processor)
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
                    conv = conversation_lib.default_conversation.copy()
                    roles = conv.roles
                    inp = input(f"{roles[0]}: ")
                    break
                except:
                    print('Please input a correct image path.')
                    continue
        if inp == 'change video':
            while True:
                try :
                    video_path = input('Please input new video path:')
                    video = load_video(
                            video_path = video_path,
                            n_frms = model.config.max_frame,
                            height = 224,
                            width = 224,
                            sampling ="uniform", return_msg = False)
                    video_tensor = video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
                    conv = conversation_lib.default_conversation.copy()
                    roles = conv.roles
                    inp = input(f"{roles[0]}: ")
                    break
                except:
                    print('Please input a correct video path.')
                    continue
        if inp == 'change sound':
            while True:
                try :
                    sound_path = input('Please input new sound path:')
                    sound = load_and_transform_audio_data([sound_path],device="cpu").to(CONFIG.device, dtype=torch.bfloat16)
                    sound_tensor = sound
                    conv = conversation_lib.default_conversation.copy()
                    roles = conv.roles
                    inp = input(f"{roles[0]}: ")
                    break
                except:
                    print('Please input a correct sound path.')
                    continue

        if not inp:
            print("exit...")
            break
        print(f"{roles[1]}: ", end="")
        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IMAGE_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * CONFIG.image_token_len + DEFAULT_IMAGE_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        elif video is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len + DEFAULT_VIDEO_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_VIDEO_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            video = None
        elif sound is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_SOUND_START_TOKEN + DEFAULT_SOUND_PATCH_TOKEN * CONFIG.sound_token_len + DEFAULT_SOUND_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_SOUND_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            sound = None
        else:
            conv.append_message(conv.roles[0], inp)

        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                videos=video_tensor,
                sounds=sound_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, 
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if image_path is not None:
            outputs = postprocess_output(outputs, image_path)
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        print(outputs)
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/ckpt/LEGO-7B")
    parser.add_argument("--image_file", type=str, default=None)
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--sound_file", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)