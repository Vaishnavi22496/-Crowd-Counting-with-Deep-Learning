import os
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from models.models import VGG19_CNN
from torch.utils.data import  DataLoader
from utils.utils import overlay_image_and_mask
from utils.data_utils import Custom_evaluation_Dataset, test_transforms



logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

now = datetime.now().strftime("%y%m%d%H%M%S")

class ModelTester(object):
    """Use the model in eval mode for testing"""

    def __init__(self, model, saved_path, test_transforms, norm=100., save=False):
        checkpoint = torch.load(saved_path)
        logging.info(f"Loading checkpoint {saved_path}")
        self.model = model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        logging.info(f"Model: {model.__class__.__name__} is ready")
        self.test_transforms = test_transforms
        self.norm = norm
        self.save = save

    def load_image(self, image_name):
        image = Image.open(image_name)
        image = self.test_transforms(image).float()
        image = image.unsqueeze(0)
        logging.debug(image.shape)
        return image

    def single_evaluate(self, image_name=None):

        image = self.load_image(image_name)
        image = image.to(device)
        preds = self.model(image)
        preds = preds.squeeze(0)
        preds = preds.data.cpu().numpy() / self.norm
        output = {
            "fname": "",
            "count": np.around(np.sum(preds)),
            "prediction": preds[0]
        }
        if self.save:
            self.save_density_map(image_name, output['prediction'], int(output['count']), image_name)
        return output

    def evaluate(self, dirname, batch_size=-1):

        data = Custom_evaluation_Dataset(dirname, transform=self.test_transforms)
        if len(data) == 0:
            logging.info(f"There are no images in {dirname}")
            return None

        batch_size = len(data) if batch_size == -1 else batch_size
        if batch_size > 5:
            logging.warn(f"You are using a batch size of {batch_size} which may result in out of memory error. We suggest to use a batch size < 5")

        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        full_preds = []
        for idx, (inputs, fname) in enumerate(loader):
            images = inputs.to(device)
            preds = self.model(images)
            preds = preds.squeeze(0)
            preds = preds.data.cpu().numpy() / self.norm
            count = np.around(np.sum(preds))
            full_preds.append({
                "fname": fname,
                "count": count,
                "prediction": preds
            })
            if batch_size > 1 and self.save:
                logging.warn("We can only save density maps at batch size = 1 for now")
            
            if batch_size == 1 and self.save:
                preds = preds.squeeze(0)
                self.save_density_map(fname[0], preds, count, fname[0])

        return full_preds

    def save_density_map(self, input_fname, denmap, count, save_fname):        
        logging.info(f"input_fname:{input_fname}, denmap: {denmap.shape}, count:{count}")
        dirname = f"output\\{now}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        overlay_image_and_mask(input_fname, denmap, count, alpha=0.4, use_img_size=True,figsize=None, dirname = dirname,save_fname =save_fname)

        logging.info(f"Saved to {dirname}")



def main(args):

    if args.model == "vggCNN":
        model = VGG19_CNN()
    else:
        raise Exception(f"Unsupported model name {args.model} provided. Unable to continue further")

    tester = ModelTester(model, args.checkpoint, test_transforms=test_transforms(), save=args.save)

    if args.single is not None:
        output = tester.single_evaluate(args.single)
        preds = output["count"]
        logging.info(f"fname: {args.single}")
        logging.info(f"People Count: {np.around(preds)}")
        logging.info("Running batch mode")
        logging.info("-" * 40)

    if args.imagepath is not None:
        output = tester.evaluate(args.imagepath, batch_size=1)
        if output is not None:
            logging.info(output[0]["fname"])
            logging.info(output[0]["count"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crowd Counter Tester")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="D:\Praveens_files\Crowd_counting_project\ckpt\\vggCNN-1e-05-0.048820221223.pt",
        help="Path of model checkpoint to continue training an existing model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vggCNN",
        help="Name of the model class"
    )
    parser.add_argument(
        "--single",
        type=str,
        default="D:\Praveens_files\Crowd_counting_project\eval_imgs\\4.jpg",
        help="Path of the single image for evaluation"
    )
    parser.add_argument(
        "--imagepath",
        type=str,
        default=None,
        help="Path of images you wish to evaluate"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="Whether to save the density map under the output folder"
    )

    args = parser.parse_args()    
    main(args)
