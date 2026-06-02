import torch

import visualisations.image_utils as iu
from visualisations.predict import predict
import utils.utils as utils
from data.data_loaders import get_augmented_data_loaders, get_augmented_data_loaders_multi_channel

def load_weights(aug, adapter, backbone, architecture, model):
    path = f"./checkpoints/best_model_{architecture}_{backbone}_{aug}_{adapter}.pth"
    state_dict = torch.load(path)
    model.load_state_dict(state_dict["model_state_dict"])

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    MODELS = utils.get_models()
    args = utils.parse_arguments()

    key = (args.mapper, args.backbone, args.architecture)
    mapper, back, arch = key
    if key not in MODELS.keys():
        keys = "\n".join(f"({adapter}, {back}, {arch})" for adapter, back, arch in MODELS.keys())
        print("Invalid combination of model  given.")
        parser.error(
            f"Invalid adapter-backbone-architecture combination given:\n\t{key}\n"
            f"Expected a combination of the following:\n{keys}"
        )

    aug = args.data_augmentation
    if aug == "aug":
        train_loader, val_loader, test_loader = get_augmented_data_loaders(batch_size=BATCH_SIZE)
        if mapper == "none":
            model = MODELS[key](21, in_channels=3).to(DEVICE)
        else:
            model = MODELS[key](21, in_channels=3, adapter=mapper).to(DEVICE)
    elif aug == "multi":
        train_loader, val_loader, test_loader = get_augmented_data_loaders_multi_channel(batch_size=BATCH_SIZE)
        model = MODELS[key](21, in_channels=7, adapter=mapper).to(DEVICE)

    load_weights(aug, mapper, back, arch, model)
    model.eval()

    mIoU_scores = []

    for image, mask in test_loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        prediction = predict(image, model)
        # for i in range(BATCH_SIZE-1):
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.subplot(1, 1, 1)
        #     plt.imshow(image[i][0:3].permute(1, 2, 0).numpy())
        #     plt.show()
        for i in range(prediction.shape[0]):
            score = utils.mIoU(prediction[i].squeeze(0), mask[i].squeeze(0), num_classes=21, ignore_index=255)
            mIoU_scores.append(score)

            # print(prediction[i].shape, torch.unique(prediction[i]))
            # print(mask[i].shape, torch.unique(mask[i]))

            # import matplotlib.pyplot as plt
            # plt.figure()

            # plt.subplot(1, 2, 1)
            # plt.title(f"mIoU score = {score}")
            # plt.imshow(iu.decode_segmap(prediction[i].numpy()))
            # plt.axis("off")

            # plt.subplot(1, 2, 2)
            # plt.title("Expected mask")
            # plt.imshow(iu.decode_segmap(mask[i].numpy()))
            # plt.axis("off")

            # plt.legend(handles=iu.create_legend(mask), bbox_to_anchor=(1.05, 1), loc="upper left")
            # plt.show()

        # mIoU_scores.append(score)

    mIoU_score = sum(mIoU_scores) / len(mIoU_scores)
    
    print(f"mIoU = {mIoU_score:.4f}")
