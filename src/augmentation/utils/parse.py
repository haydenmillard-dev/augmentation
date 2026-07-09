import argparse
from pathlib import Path

from augmentation.config.enums import Strategy

def get_augs():
    return {
        "aug": Strategy.BASIC,
        "multi": Strategy.MULTI,
    }

def strategy_type(x: str) -> Strategy:
    mapping = get_augs()
    if x not in mapping:
        raise argparse.ArgumentTypeError(f"Unknown strategy: {x}")
    return mapping[x]

def parse_arguments():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument(
        "-a", "--architecture",
        choices=["unet"],
        default="unet",
        help="Model Architecture"
    )
    # backbone
    parser.add_argument(
        "-b", "--backbone",
        choices=["none", "resnet"],
        default="none",
        help="Backbone Encoder"
    )
    # strategy
    parser.add_argument(
        '-d', '--data-augmentation',
        choices=["aug", "multi"],
        default="aug",
    )
    # parser.add_argument(
    #     "-d", "--data-augmentation",
    #     type=strategy_type,
    #     default=Strategy.BASIC,
    #     help="Data Augmentation Policy:\n\taug  - 3 channel augmentation\n\tmulti - multi-channel augmentation"
    # )
    # adapter
    parser.add_argument(
        '-m', '--mapper',
        choices=["none", "input", "deep-input", "multi-layer"],
        default="none",
        help="Adapter used to map inputs to the backbone:\n\tDefault: none: no adapter\n\tinput: a single-layer InputAdapter\n\tdeep-input: a multi-layer InputAdapter\n\tMultiLayer: a channel-grouped adapter."
    )
    # debug
    parser.add_argument(
        '--debug',
        choices=["normal", "verbose"],
        default=None,
    )
    args = parser.parse_args()

    if args.debug:
        print_arguments(args)

    return args

def DONT_parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--architecture",
        choices=["unet"],
        default="unet",
        help="Model Architecture"
    )
    parser.add_argument(
        "-b", "--backbone",
        choices=["none", "resnet"],
        default="none",
        help="Backbone Encoder"
    )
    parser.add_argument(
        '-d', '--data-augmentation',
        choices=list(Strategy),
        default=None,
        help="Data Augmentation Policy:\n\taug  - 3 channel augmentation\n\tmulti - multi-channel augmentation"
    )
    parser.add_argument(
        '-m', '--mapper',
        choices=["none", "input", "deep-input", "multi-layer"],
        default="none",
        help="Adapter used to map inputs to the backbone:\n\tDefault: none: no adapter\n\tinput: a single-layer InputAdapter\n\tdeep-input: a multi-layer InputAdapter\n\tMultiLayer: a channel-grouped adapter."
    )
    parser.add_argument(
        '--debug',
        choices=["normal", "verbose"],
        default=None,
    )
    args = parser.parse_args()

    if args.debug:
        print_arguments(args)

    return args

def check_example_arg(arg):
    try:
        return int(arg)
    except ValueError:
        pass

    path = Path(arg)
    
    valid_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    if path.is_file and path.suffix.lower() in valid_suffixes:
        return path
    
    raise argparse.ArgumentTypeError(
        f"'{value}' is neither an integer nor a valid image file."
    )

def visualise_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--strategy",
        type=strategy_type,
        default=Strategy.BASIC,
        help="Data Augmentation Policy:\n\taug  - 3 channel augmentation\n\tmulti - multi-channel augmentation"
    )
    parser.add_argument(
        '-e', '--example',
        type=check_example_arg,
        help="Integer index for an example in the entire PASCAL VOC 2012 Segmentation Dataset or a valid image file path."
    )
    parser.add_argument(
        '--debug',
        choices=["normal", "verbose"],
        default=None,
    )

    args = parser.parse_args()
    if args.debug:
        print_arguments(args)

    return args
