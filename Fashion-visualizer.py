import json
import cv2
import argparse
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


class Fashion_visualizer:

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.fashion_metadata = self.load_metadata()

    def get_parser(self):
        parser = argparse.ArgumentParser(description="--- Fashion-visualizer ---")
        parser.add_argument("--path", type=str, help="Input image path")
        parser.add_argument(
            "--th", type=float, default=0.7, help="Prediction thershold"
        )
        parser.add_argument(
            "--wt",
            type=str,
            default="config/model_v1.pth",
            help="Model weight file path",
        )
        parser.add_argument(
            "--md",
            type=str,
            default="config/fashion_metadata.json",
            help="Metadata json path",
        )
        return parser

    def load_metadata(self):
        if self.args.md:
            metadata_path = self.args.md
        else:
            raise KeyError("Metadata path error")

        with open(
            metadata_path,
            "r",
        ) as f:
            return json.load(f)

    def show_res(self, image_path, threshold, weight_path):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            threshold  # set the testing threshold for this model
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 46  # 46 classes in
        predictor = DefaultPredictor(cfg)

        im = cv2.imread(image_path)

        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=self.fashion_metadata,
            scale=1.2,
            instance_mode=ColorMode.IMAGE_BW,
        )

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        plt.figure(figsize=(8, 8))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()


def main():
    v = Fashion_visualizer()

    if v.args.path and v.args.th and v.args.wt:
        image_path = v.args.path
        threshold = v.args.th
        weight_path = v.args.wt
    else:
        raise KeyError("Input error")

    v.show_res(image_path, threshold, weight_path)


if __name__ == "__main__":
    main()
