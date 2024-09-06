from humanparsing.run_parsing import Parsing
from openpose.run_openpose import OpenPose
from utils_mask import get_mask_location
from PIL import Image


parsing_model = Parsing(0)
openpose_model = OpenPose(0)


human_img_orig = Image.open('1.jpg') 

human_img = human_img_orig.resize((768,1024))
keypoints = openpose_model(human_img.resize((384,512)))
model_parse, _ = parsing_model(human_img.resize((384,512)))
mask, mask_gray = get_mask_location('hd', "dresses", model_parse, keypoints)
mask = mask.resize((768,1024))