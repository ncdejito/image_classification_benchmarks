
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import presets
import torch
from torch import nn
import argparse
import glob
import os
import torchvision

def load_image(img_path, args):
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
    )
    img = Image.open(img_path)
    img_t = preprocessing(img)
    return torch.unsqueeze(img_t, 0)

def load_model(model_path="checkpoint.pth"):
    dict_ = torch.load(model_path)
    args = dict_['args']
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=2)
    model.load_state_dict(dict_["model"])
    model.eval()
    return model, args

def main(input_dir, output_dir):
    classes = ["1","2"]
    model, args = load_model()

    input_files = glob.glob(input_dir + "/*")
    if len(input_files) == 0:
        print("No files found")
        return 1
    for img_path in input_files:
        input = load_image(img_path, args)
        outputs = model(input)
        _, pred = torch.max(outputs, 1)
        os.rename(img_path, output_dir+"/"+classes[pred[0]]+"/"+img_path.split("/")[-1])

def get_args_parser(add_help=True):    
    parser = argparse.ArgumentParser(description="PyTorch Classification Inference", add_help=add_help)
    parser.add_argument("--input_dir", default="example_directory_structure/input", type=str, help="input directory")
    parser.add_argument("--output_dir", default="example_directory_structure/output", type=str, help="output directory")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args.input_dir, args.output_dir)
