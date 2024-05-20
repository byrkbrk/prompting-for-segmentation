from argparse import ArgumentParser, ArgumentTypeError
from segmentation import PromptSAM



def parse_tuple(s):
    """Returns tuple of integers obtained from given string s"""
    try:
        return tuple(map(int, s.split(",")))
    except ValueError:
        raise ArgumentTypeError("Tuples must be integers seperated by comma")
    
def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Segment given image")
    parser.add_argument("image_name", type=str, default=None, help="Name of the image that be processed")
    parser.add_argument("point_or_bbox_prompts", nargs="+", type=parse_tuple, help="List of point prompts in (height, width) or bbox prompts in (x_min, y_min, x_max, y_max)")
    parser.add_argument("--label_prompts", nargs="+", type=int, default=None, help="List of labels of point prompts")
    parser.add_argument("--image_size", nargs="+", type=int, default=[1024, 1024], help="Size (height, width) to which the image be transformed")
    parser.add_argument("--checkpoint_name", type=str, default="FastSAM-x.pt", choices=["FastSAM-x.pt", "FastSAM-s.pt"], help="Name of the pretrained model for FastSAM")
    parser.add_argument("--device", type=str, default=None, help="Name of the device on which the model be run")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    PromptSAM(args.image_name, 
              args.checkpoint_name,
              args.device).segment(args.point_or_bbox_prompts, 
                                   args.label_prompts, 
                                   args.image_size)