import cv2
from matplotlib import pyplot as plt
import time
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from ptyolact.yolact import Yolact
from ptyolact.data import COLORS
from ptyolact.utils.augmentations import FastBaseTransform
from ptyolact.layers.output_utils import postprocess, undo_image_transformation
from ptyolact.utils import timer
from ptyolact.utils.functions import SavePath
from ptyolact.data import cfg, set_cfg

color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    # If the config uses lincomb mask, output a visualization of how those masks are created
    args_display_lincomb = False
    # Do crop output masks with the predicted bounding box.
    args_crop = True
    # Detections with a score under this threshold will not be considerd. This currently only works in display mode.
    args_score_threshold = 0
    # Further restrict the number of predictions to parse
    args_top_k = 5
    # Whether or not to display masks over bounding boxes
    args_display_masks = True
    # When displaying / saving video, draw the FPS on the frame
    args_display_fps = False
    # Whether or not to display text (class [score])
    args_display_text = True
    # Whether or not to display bboxes around masks
    args_display_bboxes = True
    # Wheter or not to display scores in addition to classes
    args_display_scores = True

    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        # postprocess output -> classes, scores, boxes, masks
        t = postprocess(dets_out, w, h, visualize_lincomb=args_display_lincomb,
                                        crop_masks       =args_crop,
                                        score_threshold  =args_score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args_top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args_top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args_score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args_display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)

        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args_display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args_display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if args_display_text or args_display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args_display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args_display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args_display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return img_numpy


def evalimage(net:Yolact, path:str, save_path:str=None):
    # Whether to use a faster, but not entirely correct version of NMS
    net.detect.use_fast_nms = True
    # Whether compute NMS cross-class or per-class
    net.detect.use_cross_class_nms = False
    # Outputs stuff for scripts/compute_mask.py
    cfg.mask_proto_debug = False

    img = cv2.imread(path)
    frame = torch.from_numpy(img).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    st = time.time()
    preds = net(batch)
    for key in preds[0]['detection']:
        print(key, preds[0]['detection'][key].shape)
    print(f"[INFO] YOLACT prediction time: {time.time() - st}")

    # h, w, _ = frame.shape
    # classes, scores, boxes, masks = yolact_postprocess(preds, frame, top_k)
    #
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("Query")
    # plt.show()
    # for i in range(classes.shape[0]):
    #     _class = cfg.dataset.class_names[classes[i]]
    #     x1, y1, x2, y2 = boxes[i]
    #     _mask = np.zeros_like(masks[i])
    #     # _mask[y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    #
    #     plt.imshow(masks[i], cmap='gray')
    #     plt.title(_class)
    #     plt.show()

    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)


def detect(net:Yolact, path:str, top_k:int=5, cuda:bool=True):
    """

    :param net: Yolact 모델
    :param path: 이미지 경로
    :param top_k: detection된 정보 중 score기반으로 상위 몇 개를 선택할지

    :return: classes, scores, boxes, masks, p3_out(이미지 검색에 사용되는 feature map)
    """

    # 이미지 불러오기
    img = cv2.imread(path)
    frame = torch.from_numpy(img).float()
    if cuda:
        frame = frame.to(torch.device("cuda:0"))
    batch = FastBaseTransform()(frame.unsqueeze(0))

    # YOLACT 실행
    st = time.time()
    # with torch.autograd.profiler.profile(record_shapes=True, use_cuda=cuda) as prof:
    det_outs = net(batch)
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
    # print(f"[INFO] YOLACT prediction time: {time.time() - st}")

    # 이미지 검색을 위한 feature 처리
    fpn_feature = det_outs[0]['detection']['fpn_feature']
    global_avg_pool_out = F.adaptive_avg_pool2d(fpn_feature, (1, 1))
    global_avg_pool_out = global_avg_pool_out.view(1, global_avg_pool_out.size(1))

    # YOLACT output 데이터 후처리
    # YOLACT에서 나온 값들은 바로 사용할 수가 없음
    # 특히 masks 값의 경우 coefficient값이기 때문에 후처리를 통해
    # 사용가능한 mask 형태로 바꿔줘야 함
    h, w, _ = frame.shape
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        # postprocess output -> classes, scores, boxes, masks
        t = postprocess(det_outs, w, h,
                        visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=0.1)
        cfg.rescore_bbox = save


    with timer.env('Copy'):
        # top_k개만 추출
        idx = t[1].argsort(0, descending=True)[:top_k]
        masks = t[3][idx].cpu().numpy()
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    return classes, scores, boxes, masks, global_avg_pool_out

if __name__ == "__main__":

    import os
    image_dir = "../images"
    image_names = ['test12.jpg']
    trained_model = 'weights/yolact_plus_resnet50_54_800000.pth'

    # Yolact config Setting
    model_path = SavePath.from_str(trained_model)
    config = model_path.model_name + '_config'
    print(f'Config not specified. Parsed {config} from the file name.')
    set_cfg(config)

    # GPU 사용 여부
    cuda = True
    print('Loading model... ', end='')
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    print(' Done.')

    # Whether to use a faster, but not entirely correct version of NMS
    net.detect.use_fast_nms = True
    # Whether compute NMS cross-class or per-class
    net.detect.use_cross_class_nms = False
    # Outputs stuff for scripts/compute_mask.py
    cfg.mask_proto_debug = False

    if cuda:
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if cuda:
        net = net.cuda()

    with torch.no_grad():

        for image_name in image_names:
            image_path = f"{image_dir}/{image_name}"
            if not os.path.exists(image_path):
                print(f"doesn't exist {image_path}")
                continue
            else:
                print(f"-------Test..{image_name}---------")

            # 이미지에 Yolact 결과를 출력하는 부분
            # evalimage(net, image_path)

            # pytorch에서 처음 모델을 돌리면 대략 1초정도가 소모되는 경향이 있어
            # 미리 한 번 볼리는 코드
            timer.disable_all()
            detect(net, image_path, cuda=cuda)
            timer.enable_all()

            # mask 부분만 따로 보는 코드
            st = time.time()
            classes, scores, boxes, masks, fpn_feature = detect(net, image_path, top_k=15, cuda=cuda)
            print(f"Detect time {time.time() - st} sec...")

            plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            plt.title("Query")
            plt.show()
            for i in range(classes.shape[0]):
                _class = cfg.dataset.class_names[classes[i]]
                x1, y1, x2, y2 = boxes[i]
                _mask = np.zeros_like(masks[i])
                # _mask[y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]

                plt.imshow(masks[i], cmap='gray')
                plt.title(_class)
                plt.show()

        # 소모된 시간
        timer.print_stats()


