from PIL import Image
from io import BytesIO
import base64
import torch
import re
import os 
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from transformers import StoppingCriteria
from lego.constants import IMAGE_TOKEN_INDEX

colors = ["#0000FF","#FF0000","#00FF00","#FFFF00","#00FFFF","#FF00FF","#800080","#FFA500","#008000","#A52A2A","#FFC0CB","#00CED1","#8B008B","#FFD700","#7FFFD4","#FF4500","#2E8B57","#800000","#8A2BE2","#FF1493"]

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def load_image_square(image_file, image_processor,image_aspect_ratio='pad'):
    image = Image.open(image_file).convert('RGB')
    if image_aspect_ratio == 'pad':
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    return image

def postprocess_box(box, ori_w, ori_h):
    if ori_w == ori_h:
        return box
    if ori_w > ori_h:
        x1, y1, x2, y2 = box
        y1 -= (ori_w - ori_h) // 2
        y2 -= (ori_w - ori_h) // 2
        box = x1, y1, x2, y2
        return box
    x1, y1, x2, y2 = box
    x1 -= (ori_h - ori_w) // 2
    x2 -= (ori_h - ori_w) // 2
    box = x1, y1, x2, y2
    return box

def postprocess_output(outputs, image_file):
    if image_file is None:
        return
    image = Image.open(image_file).convert('RGB')
    ori_width, ori_height = image.size
    max_ori = max(ori_width, ori_height)
    regex = re.compile(r'\[[0-9.]+, [0-9.]+, [0-9.]+, [0-9.]+\]')
    matches = re.findall(regex, outputs)
    new_outputs = outputs
    for match in matches:
        try:
            x1, y1, x2, y2 = [float(part.strip()) for part in match.strip('[]').split(',')]
            square_bbox = round(x1 * max_ori), round(y1 * max_ori), round(x2 * max_ori), round(y2 * max_ori)
            ori_box = postprocess_box(square_bbox, ori_width, ori_height)
            new_x1, new_y1, new_x2, new_y2 = [round(ori_box[0]/ori_width,3), round(ori_box[1]/ori_height,3), round(ori_box[2]/ori_width,3), round(ori_box[3]/ori_height,3)]
            new_x1 = np.clip(new_x1, 0, 1)
            new_y1 = np.clip(new_y1, 0, 1)
            new_x2 = np.clip(new_x2, 0, 1)
            new_y2 = np.clip(new_y2, 0, 1)
            new_bbox_string = f'[{new_x1}, {new_y1}, {new_x2}, {new_y2}]'
            new_outputs = new_outputs.replace(match, new_bbox_string)
        except:
            pass
    return new_outputs

def postprocess_image_answer_gradio(description, image_path):
    coordinates_groups = []
    start_index = description.find("[")
    tmp_index = start_index
    while start_index != -1:
        end_index = description.find("]", tmp_index)
        if end_index != -1:
            if description[end_index+1]=='[':
                tmp_index = end_index+1
                continue
            tmp_index = end_index
            coordinates_groups.append(description[start_index: tmp_index+1])
            start_index = description.find("[", end_index)
            tmp_index = start_index
        else:
            break
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(image)
    width, height = image.size
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    for i,coordinates_group in enumerate(coordinates_groups):
        color = colors[i]
        replace_string = f'[{i}]'
        description = description.replace(coordinates_group, f"<span style='color: {color};'>{replace_string}</span>")
        pattern = r'\[[^\[\]]+\]'
        matches = re.findall(pattern, coordinates_group)
        for coord in matches:
            coord = [float(c) for c in coord[1:-1].split(",")]
            x_min, y_min, x_max, y_max = coord
            x_min_abs, y_min_abs = x_min * width, y_min * height
            x_max_abs, y_max_abs = x_max * width, y_max * height
            width_abs = x_max_abs - x_min_abs
            height_abs = y_max_abs - y_min_abs
            rect = patches.Rectangle(
                (x_min_abs, y_min_abs),
                width_abs,
                height_abs,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()
    return description,filename

def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
