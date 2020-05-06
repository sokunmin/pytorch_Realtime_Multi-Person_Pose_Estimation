import sys
sys.path.append('.')
import cv2
import argparse
import numpy as np
import torch

from lib.network.rtpose_vgg import get_model
from evaluate.coco_eval import get_outputs
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp, paf_to_pose
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='weights/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

test_image = './readme/p2.jpg'
oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image

with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
          
print(im_scale)
# > Python
humans = paf_to_pose(heatmap, paf, cfg)
# > C++
humans = paf_to_pose_cpp(heatmap, paf, cfg)

out = draw_humans(oriImg, humans)
cv2.imwrite('result.png',out)   

