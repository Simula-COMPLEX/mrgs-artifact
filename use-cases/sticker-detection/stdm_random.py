import cv2
import csv
import json
import random
import numpy as np
import torch
import torchvision
from pathlib import Path

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead, det_utils
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import os


# Paths

PROJECT_ROOT = Path.cwd()

IMAGES_DIR = PROJECT_ROOT / "data/valid"
ANNOTATIONS_PATH = PROJECT_ROOT / "data/valid/annotations.coco.json"

# Select one of the detectors trained in the project.
MODEL_NAME = "ssd300_vgg16"

MODEL_CHECKPOINTS = {
    "ssd300_vgg16": "epoch=33-step=4114.ckpt",
    "ssdlite320_mobilenet_v3_large": "epoch=18-step=9139.ckpt",
}

if MODEL_NAME not in MODEL_CHECKPOINTS:
    raise ValueError(
        f"Unsupported MODEL_NAME: {MODEL_NAME}. "
        f"Choose from: {sorted(MODEL_CHECKPOINTS)}"
    )

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / MODEL_NAME
    / MODEL_CHECKPOINTS[MODEL_NAME]
)

CATEGORY = "valid"

# Keep the output separate from the dataset and from other results.
OUT_ROOT = (
    PROJECT_ROOT
    / "random_jaccard_sticker"
    / MODEL_NAME
    / CATEGORY
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)


RUN_ID = os.environ.get("RUN_ID", "")
if RUN_ID:
    OUT_ROOT = OUT_ROOT / RUN_ID

OUT_ROOT.mkdir(parents=True, exist_ok=True)

POP_SIZE = 40
N_GEN = 250
BASE_SEED = int(os.environ.get("BASE_SEED", 42))

# Faster R-CNN inference settings.
#
# These values are applied identically to the original and
# perturbed images. Image resizing is not configured here:
# Torchvision uses the same internal defaults used during training.
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.50
TOP_K = 5000

# None selects CUDA when available, otherwise CPU.
# Explicit examples: "cuda:0" or "cpu".
DEVICE = os.environ.get("DEVICE", None)

# Matching threshold between original and perturbed predictions.
PRED_MATCH_IOU_THRESHOLD = 0.5

# Search vector:
# x = [brightness_delta, contrast_factor, gamma_value, blur_strength]
BRIGHTNESS_MIN, BRIGHTNESS_MAX = -40.0, 40.0
CONTRAST_MIN, CONTRAST_MAX = 0.6, 1.6
GAMMA_MIN, GAMMA_MAX = 0.5, 2.0
BLUR_MIN, BLUR_MAX = 0.0, 15.0

OBJECTIVE_NAMES = [
    "one_minus_mr_score",
    "perturbation_magnitude",
]

# Both objectives are expected to be in [0, 1].
# The reference point is slightly worse than the worst possible values.
HV_REF_POINT = np.array([1.1, 1.1], dtype=float)



# General utilities

def safe_name(path: Path) -> str:
    return path.stem.replace("/", "_").replace(" ", "_")


def compute_iou(box_a, box_b) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )
    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def save_csv(path: Path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def json_safe(obj):
    """
    Convert NumPy values and non-finite floating-point values into
    JSON-compatible values.
    """
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return [json_safe(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return [json_safe(value) for value in obj.tolist()]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value

    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj

    return obj


def get_class_name(class_names, class_id: int) -> str:
    """
    Get a readable class name from the COCO category mapping.
    """
    return str(class_names.get(class_id, class_id))


def resolve_device():
    if DEVICE is not None:
        return torch.device(DEVICE)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")



# COCO ground-truth loading
#
# Ground truth is used only for visualization and metadata.
# It is not used in the MR oracle, search objectives, selection,
# or hypervolume calculation.

def load_coco_dataset(annotation_path: Path):
    """
    Load the COCO annotation file once.

    Returns:
      - image information indexed by file name;
      - annotations indexed by image ID;
      - class-name mapping;
      - number of model classes including background.
    """
    with open(annotation_path, "r") as file:
        coco_data = json.load(file)

    image_info_by_name = {
        image_info["file_name"]: image_info
        for image_info in coco_data.get("images", [])
    }

    annotations_by_image_id = {}

    for annotation in coco_data.get("annotations", []):
        image_id = annotation["image_id"]

        annotations_by_image_id.setdefault(
            image_id,
            [],
        ).append(annotation)

    class_names = {
        int(category["id"]): str(category["name"])
        for category in coco_data.get("categories", [])
    }

    if not class_names:
        raise RuntimeError(
            f"No categories found in: {annotation_path}"
        )

    # Faster R-CNN reserves label 0 for background.
    num_classes = max(class_names.keys()) + 1

    return (
        image_info_by_name,
        annotations_by_image_id,
        class_names,
        num_classes,
    )


def load_coco_gt_annotations(
    img_path: Path,
    image_info_by_name,
    annotations_by_image_id,
    image_width: int,
    image_height: int,
):
    """
    Load COCO-format annotations for one image.

    COCO bbox format:
        [x, y, width, height]

    Boxes are returned in the same common structure used by
    the screw-detection script.
    """
    boxes = []

    image_info = image_info_by_name.get(img_path.name)

    if image_info is None:
        print(
            f"[warn] Missing COCO image entry: "
            f"{img_path.name}"
        )
        return boxes

    image_id = image_info["id"]

    for annotation in annotations_by_image_id.get(
        image_id,
        [],
    ):
        bbox = annotation.get("bbox")

        if bbox is None or len(bbox) != 4:
            print(
                f"[warn] Invalid COCO bbox for "
                f"annotation {annotation.get('id')}: {bbox}"
            )
            continue

        class_id = int(annotation["category_id"])
        x, y, width, height = map(float, bbox)

        x1 = float(np.clip(
            x,
            0.0,
            max(image_width - 1, 0),
        ))
        y1 = float(np.clip(
            y,
            0.0,
            max(image_height - 1, 0),
        ))
        x2 = float(np.clip(
            x + width,
            0.0,
            max(image_width - 1, 0),
        ))
        y2 = float(np.clip(
            y + height,
            0.0,
            max(image_height - 1, 0),
        ))

        boxes.append({
            "class_id": int(class_id),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [
                x1,
                y1,
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
            ],
        })

    return boxes



# Torchvision detector loading and inference

def build_torchvision_detector(
    model_name: str,
    num_classes: int,
):
    """
    Reconstruct one of the detector architectures used during training.

    No explicit image-size parameters are introduced. Each detector uses
    the same internal Torchvision preprocessing associated with its
    architecture and used by the original training implementation.
    """
    if model_name == "fasterrcnn_resnet50_fpn":
        model = (
            torchvision.models.detection
            .fasterrcnn_resnet50_fpn(
                weights=None,
                trainable_backbone_layers=5,
            )
        )

        in_features = (
            model.roi_heads.box_predictor
            .cls_score.in_features
        )

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes,
        )

    elif model_name == "fasterrcnn_resnet50_fpn_v2":
        model = (
            torchvision.models.detection
            .fasterrcnn_resnet50_fpn_v2(
                weights=None,
                trainable_backbone_layers=5,
            )
        )

        in_features = (
            model.roi_heads.box_predictor
            .cls_score.in_features
        )

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes,
        )

    elif model_name == "retinanet_resnet50_fpn":
        model = (
            torchvision.models.detection
            .retinanet_resnet50_fpn(
                weights=None,
                trainable_backbone_layers=5,
            )
        )

        num_anchors = (
            model.head.classification_head.num_anchors
        )
        out_channels = model.backbone.out_channels

        model.head.classification_head = (
            RetinaNetClassificationHead(
                in_channels=out_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
            )
        )

    elif model_name == "retinanet_resnet50_fpn_v2":
        model = (
            torchvision.models.detection
            .retinanet_resnet50_fpn_v2(
                weights=None,
                trainable_backbone_layers=5,
            )
        )

        num_anchors = (
            model.head.classification_head.num_anchors
        )
        out_channels = model.backbone.out_channels

        model.head.classification_head = (
            RetinaNetClassificationHead(
                in_channels=out_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
            )
        )

    elif model_name == "ssd300_vgg16":
        model = (
            torchvision.models.detection
            .ssd300_vgg16(
                weights=None,
                trainable_backbone_layers=5,
            )
        )

        out_channels = det_utils.retrieve_out_channels(
            model.backbone,
            (300, 300),
        )
        num_anchors = (
            model.anchor_generator
            .num_anchors_per_location()
        )

        model.head.classification_head = (
            SSDClassificationHead(
                out_channels,
                num_anchors,
                num_classes,
            )
        )

    elif model_name == "ssdlite320_mobilenet_v3_large":
        model = (
            torchvision.models.detection
            .ssdlite320_mobilenet_v3_large(
                weights=None,
                trainable_backbone_layers=6,
            )
        )

        out_channels = det_utils.retrieve_out_channels(
            model.backbone,
            (320, 320),
        )
        num_anchors = (
            model.anchor_generator
            .num_anchors_per_location()
        )

        model.head.classification_head = (
            SSDClassificationHead(
                out_channels,
                num_anchors,
                num_classes,
            )
        )

    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}"
        )

    return model


def configure_detector_inference(
    model,
    model_name: str,
):
    """
    Apply the same score, NMS, and maximum-detection settings to the
    original and perturbed images.

    The appropriate attributes differ slightly across detector families.
    """
    if model_name.startswith("fasterrcnn_"):
        model.roi_heads.score_thresh = SCORE_THRESHOLD
        model.roi_heads.nms_thresh = NMS_THRESHOLD
        model.roi_heads.detections_per_img = TOP_K

    elif model_name.startswith("retinanet_"):
        model.score_thresh = SCORE_THRESHOLD
        model.nms_thresh = NMS_THRESHOLD
        model.detections_per_img = TOP_K
        model.topk_candidates = TOP_K

    elif (
        model_name == "ssd300_vgg16"
        or model_name == "ssdlite320_mobilenet_v3_large"
    ):
        model.score_thresh = SCORE_THRESHOLD
        model.nms_thresh = NMS_THRESHOLD
        model.detections_per_img = TOP_K
        model.topk_candidates = TOP_K

    else:
        raise ValueError(
            f"Unsupported model architecture: {model_name}"
        )


def load_torchvision_model(
    model_name: str,
    model_path: Path,
    num_classes: int,
    device,
):
    """
    Build the selected architecture and load its Lightning checkpoint.
    """
    model = build_torchvision_detector(
        model_name=model_name,
        num_classes=num_classes,
    )

    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    # The LightningModule stores the detector as self.model,
    # so checkpoint keys normally begin with "model.".
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        if cleaned_key.startswith("model."):
            cleaned_key = cleaned_key[len("model."):]

        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    configure_detector_inference(
        model=model,
        model_name=model_name,
    )

    model.to(device)
    model.eval()

    return model


def run_torchvision_detector(model, image, device):
    """
    Run any supported Torchvision detector and return predictions in
    the common structure used by the MR implementation.

    All supported models return:
        boxes, labels, and scores.
    """
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image_tensor = torch.from_numpy(
        image_rgb
    ).permute(2, 0, 1).float() / 255.0

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        outputs = model([image_tensor])

    predictions = []

    if not outputs:
        return predictions

    output = outputs[0]

    boxes = output["boxes"].detach().cpu().numpy()
    confidences = output["scores"].detach().cpu().numpy()
    class_ids = (
        output["labels"]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    for box, confidence, class_id in zip(
        boxes,
        confidences,
        class_ids,
    ):
        if float(confidence) < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box)

        predictions.append({
            "class_id": int(class_id),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [
                x1,
                y1,
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
            ],
            "confidence": float(confidence),
        })

        if len(predictions) >= TOP_K:
            break

    return predictions



# One-to-one class-aware matching for the MR oracle

def match_predictions_to_reference(
    ref_preds,
    pert_preds,
    iou_threshold=0.5,
):
    """
    Perform greedy one-to-one matching between original/reference
    predictions and perturbed predictions.

    A reference prediction is preserved only when:
      1. the perturbed prediction has the same predicted class; and
      2. their bounding-box IoU is at least iou_threshold.

    A class change is therefore represented as:
      - one lost reference prediction; and
      - one new perturbed prediction.
    """
    matches = []
    used_pert = set()

    for ref_idx, ref in enumerate(ref_preds):
        best_iou = 0.0
        best_pert_idx = None

        for pert_idx, pert in enumerate(pert_preds):
            if pert_idx in used_pert:
                continue

            if int(ref["class_id"]) != int(pert["class_id"]):
                continue

            iou = compute_iou(
                ref["bbox_xyxy"],
                pert["bbox_xyxy"],
            )

            if iou > best_iou:
                best_iou = iou
                best_pert_idx = pert_idx

        if best_pert_idx is not None and best_iou >= iou_threshold:
            used_pert.add(best_pert_idx)

            matched_pert = pert_preds[best_pert_idx]

            matches.append({
                "ref_idx": int(ref_idx),
                "pert_idx": int(best_pert_idx),
                "matched": True,
                "class_id": int(ref["class_id"]),
                "iou": float(best_iou),
                "ref_confidence": float(ref["confidence"]),
                "pert_confidence": float(matched_pert["confidence"]),
            })
        else:
            matches.append({
                "ref_idx": int(ref_idx),
                "pert_idx": None,
                "matched": False,
                "class_id": int(ref["class_id"]),
                "iou": 0.0,
                "ref_confidence": float(ref["confidence"]),
                "pert_confidence": 0.0,
            })

    return matches, used_pert



# Perturbation operators

def apply_illumination_and_blur(image, x):
    """
    Apply the same global perturbation sequence used in the
    face and screw cases:

      1. contrast and brightness;
      2. gamma correction;
      3. Gaussian blur.
    """
    brightness_delta, contrast_factor, gamma_value, blur_strength = x

    image_float = image.astype(np.float32)

    # Brightness and contrast.
    image_float = image_float * contrast_factor + brightness_delta
    image_float = np.clip(image_float, 0, 255)

    # Gamma correction.
    image_normalized = image_float / 255.0
    image_gamma = np.power(image_normalized, gamma_value) * 255.0
    output = np.clip(image_gamma, 0, 255).astype(np.uint8)

    # Gaussian blur.
    if blur_strength >= 1.0:
        kernel_size = int(round(blur_strength))

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel_size = max(3, kernel_size)

        output = cv2.GaussianBlur(
            output,
            (kernel_size, kernel_size),
            0,
        )

    return output


def perturbation_magnitude_image_space(orig_image, pert_image) -> float:
    """
    Normalized 95th-percentile absolute pixel difference.
    """
    difference = np.abs(
        orig_image.astype(np.float32)
        - pert_image.astype(np.float32)
    )

    magnitude = np.percentile(difference, 95) / 255.0

    return float(np.clip(magnitude, 0.0, 1.0))



# MR metrics

def compute_mr_metrics_from_predictions(ref_preds, pert_preds):
    """
    Label-free MR score.

    Reference:
        selected Torchvision detector predictions on the original image.

    Follow-up:
        selected Torchvision detector predictions on the perturbed image.

    MR1: Detection preservation
        Jaccard distance over class-aware matched, lost, and new
        prediction sets.

    MR2: Confidence stability
        Computed only over preserved predictions.

    MR3: Localization stability
        Computed only over preserved predictions.

    Final MR score:
        If at least one prediction is preserved:
            mean(MR1, MR2, MR3)
        Otherwise:
            MR1 only
    """
    num_reference = len(ref_preds)
    num_perturbed = len(pert_preds)

    if num_reference == 0:
        return {
            "num_reference_predictions": 0,
            "num_perturbed_predictions": int(num_perturbed),
            "matched_after": 0,
            "lost_predictions": 0,
            "new_predictions": int(num_perturbed),
            "detection_jaccard_similarity": 0.0,
            "detected_ratio": 0.0,
            "confidence_ratio": None,
            "mean_iou": None,
            "detection_violation": 1.0,
            "confidence_violation": None,
            "localization_violation": None,
            "mr_score": 1.0,
        }

    matches, used_pert = match_predictions_to_reference(
        ref_preds=ref_preds,
        pert_preds=pert_preds,
        iou_threshold=PRED_MATCH_IOU_THRESHOLD,
    )

    preserved_matches = [
        match for match in matches if match["matched"]
    ]
    matched_after = len(preserved_matches)

    lost_predictions = num_reference - matched_after
    new_predictions = num_perturbed - len(used_pert)

    union_size = (
        matched_after
        + lost_predictions
        + new_predictions
    )

    if union_size == 0:
        detection_jaccard_similarity = 1.0
    else:
        detection_jaccard_similarity = (
            matched_after / union_size
        )

    detection_violation = (
        1.0 - detection_jaccard_similarity
    )

    detected_ratio = matched_after / num_reference

    if matched_after == 0:
        confidence_ratio = None
        mean_iou = None
        confidence_violation = None
        localization_violation = None

        mr_score = detection_violation

    else:
        confidence_ratios = []
        matched_ious = []

        for match in preserved_matches:
            confidence_ratio_value = (
                match["pert_confidence"]
                / max(match["ref_confidence"], 1e-9)
            )

            # Preserve the directional confidence logic:
            # confidence decreases are penalized, increases are not.
            confidence_ratio_value = float(
                np.clip(confidence_ratio_value, 0.0, 1.0)
            )

            confidence_ratios.append(confidence_ratio_value)
            matched_ious.append(
                float(np.clip(match["iou"], 0.0, 1.0))
            )

        confidence_ratio = float(
            np.mean(confidence_ratios)
        )
        mean_iou = float(np.mean(matched_ious))

        confidence_violation = 1.0 - confidence_ratio
        localization_violation = 1.0 - mean_iou

        mr_score = float(np.mean([
            detection_violation,
            confidence_violation,
            localization_violation,
        ]))

    return {
        "num_reference_predictions": int(num_reference),
        "num_perturbed_predictions": int(num_perturbed),
        "matched_after": int(matched_after),
        "lost_predictions": int(lost_predictions),
        "new_predictions": int(new_predictions),
        "detection_jaccard_similarity": float(
            detection_jaccard_similarity
        ),
        "detected_ratio": float(detected_ratio),
        "confidence_ratio": (
            None
            if confidence_ratio is None
            else float(confidence_ratio)
        ),
        "mean_iou": (
            None if mean_iou is None else float(mean_iou)
        ),
        "detection_violation": float(detection_violation),
        "confidence_violation": (
            None
            if confidence_violation is None
            else float(confidence_violation)
        ),
        "localization_violation": (
            None
            if localization_violation is None
            else float(localization_violation)
        ),
        "mr_score": float(mr_score),
    }



# Multi-objective selection utilities

def calc_crowding_distance(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    num_solutions, num_objectives = objectives.shape

    if num_solutions == 0:
        return np.array([], dtype=float)

    if num_solutions <= 2:
        return np.full(num_solutions, np.inf, dtype=float)

    crowding_distance = np.zeros(
        num_solutions,
        dtype=float,
    )

    for objective_idx in range(num_objectives):
        sorted_indices = np.argsort(
            objectives[:, objective_idx]
        )
        sorted_values = objectives[
            sorted_indices,
            objective_idx,
        ]

        crowding_distance[sorted_indices[0]] = np.inf
        crowding_distance[sorted_indices[-1]] = np.inf

        minimum = sorted_values[0]
        maximum = sorted_values[-1]
        denominator = maximum - minimum

        if denominator < 1e-12:
            continue

        for position in range(1, num_solutions - 1):
            solution_idx = sorted_indices[position]

            if np.isinf(crowding_distance[solution_idx]):
                continue

            previous_value = sorted_values[position - 1]
            next_value = sorted_values[position + 1]

            crowding_distance[solution_idx] += (
                (next_value - previous_value)
                / denominator
            )

    return crowding_distance


def select_rank_fill_with_crowding(
    F_pop,
    X_pop,
    n_select,
):
    """
    Select n_select solutions using:
      1. non-dominated rank;
      2. crowding distance when the final accepted front does not fit.
    """
    F_pop = np.asarray(F_pop, dtype=float)
    X_pop = np.asarray(X_pop, dtype=float)

    if len(F_pop) != len(X_pop):
        raise ValueError(
            "F_pop and X_pop must contain the same number of rows."
        )

    if n_select > len(F_pop):
        raise ValueError(
            "n_select cannot exceed the population size."
        )

    non_dominated_sorting = NonDominatedSorting()
    fronts, ranks = non_dominated_sorting.do(
        F_pop,
        return_rank=True,
    )

    selected_indices = []

    for front in fronts:
        front = list(front)

        if len(selected_indices) >= n_select:
            break

        remaining = n_select - len(selected_indices)

        if len(front) <= remaining:
            selected_indices.extend(front)
        else:
            crowding_distance = calc_crowding_distance(
                F_pop[front]
            )

            descending_order = np.argsort(
                -crowding_distance
            )

            selected_from_front = [
                front[index]
                for index in descending_order[:remaining]
            ]

            selected_indices.extend(
                selected_from_front
            )

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    return (
        X_pop[selected_indices],
        F_pop[selected_indices],
        ranks[selected_indices],
    )


def get_rank0_indices(F):
    F = np.asarray(F, dtype=float)

    if F.size == 0:
        return np.array([], dtype=int)

    non_dominated_sorting = NonDominatedSorting()

    indices = non_dominated_sorting.do(
        F,
        only_non_dominated_front=True,
    )

    return np.asarray(indices, dtype=int)


def get_all_ranks(F):
    F = np.asarray(F, dtype=float)

    if F.size == 0:
        return np.array([], dtype=int)

    non_dominated_sorting = NonDominatedSorting()
    _, ranks = non_dominated_sorting.do(
        F,
        return_rank=True,
    )

    return np.asarray(ranks, dtype=int)


def compute_hv(F, hv_indicator) -> float:
    F = np.asarray(F, dtype=float)

    if F.size == 0:
        return 0.0

    clipped = np.clip(
        F,
        0.0,
        HV_REF_POINT,
    )

    return float(hv_indicator(clipped))



# Visualization and candidate saving

def draw_predictions(
    image,
    gt_boxes,
    predictions,
    class_names,
    output_path,
):
    visualization = image.copy()

    # Ground truth in blue.
    for gt in gt_boxes:
        x1, y1, x2, y2 = map(
            int,
            gt["bbox_xyxy"],
        )
        class_id = int(gt["class_id"])
        class_name = get_class_name(
            class_names,
            class_id,
        )

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            visualization,
            f"GT {class_name}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
        )

    # Predictions in green.
    for prediction in predictions:
        x1, y1, x2, y2 = map(
            int,
            prediction["bbox_xyxy"],
        )
        class_id = int(prediction["class_id"])
        class_name = get_class_name(
            class_names,
            class_id,
        )
        confidence = float(prediction["confidence"])

        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            visualization,
            f"{class_name} {confidence:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        visualization,
    )


def evaluate_candidate(
    x,
    orig_image,
    ref_preds,
    model,
    device,
):
    perturbed_image = apply_illumination_and_blur(
        orig_image,
        x,
    )

    perturbed_predictions = run_torchvision_detector(
        model,
        perturbed_image,
        device,
    )

    mr_metrics = compute_mr_metrics_from_predictions(
        ref_preds,
        perturbed_predictions,
    )

    perturbation_magnitude = (
        perturbation_magnitude_image_space(
            orig_image,
            perturbed_image,
        )
    )

    detection_binary = (
        mr_metrics["detection_violation"] > 0
    )

    confidence_binary = (
        mr_metrics["confidence_violation"] is not None
        and mr_metrics["confidence_violation"] > 0
    )

    localization_binary = (
        mr_metrics["localization_violation"] is not None
        and mr_metrics["localization_violation"] > 0
    )

    mr_score_binary = mr_metrics["mr_score"] > 0

    (
        brightness_delta,
        contrast_factor,
        gamma_value,
        blur_strength,
    ) = x

    row = {
        "brightness_delta": float(brightness_delta),
        "contrast_factor": float(contrast_factor),
        "gamma_value": float(gamma_value),
        "blur_strength": float(blur_strength),
        "perturbation_magnitude": float(
            perturbation_magnitude
        ),
        "mr_score": float(mr_metrics["mr_score"]),
        "num_reference_predictions": int(
            mr_metrics["num_reference_predictions"]
        ),
        "num_perturbed_predictions": int(
            mr_metrics["num_perturbed_predictions"]
        ),
        "matched_after": int(
            mr_metrics["matched_after"]
        ),
        "lost_predictions": int(
            mr_metrics["lost_predictions"]
        ),
        "new_predictions": int(
            mr_metrics["new_predictions"]
        ),
        "detection_jaccard_similarity": float(
            mr_metrics["detection_jaccard_similarity"]
        ),
        "detected_ratio": float(
            mr_metrics["detected_ratio"]
        ),
        "confidence_ratio": (
            None
            if mr_metrics["confidence_ratio"] is None
            else float(mr_metrics["confidence_ratio"])
        ),
        "mean_iou": (
            None
            if mr_metrics["mean_iou"] is None
            else float(mr_metrics["mean_iou"])
        ),
        "detection_violation": float(
            mr_metrics["detection_violation"]
        ),
        "confidence_violation": (
            None
            if mr_metrics["confidence_violation"] is None
            else float(mr_metrics["confidence_violation"])
        ),
        "localization_violation": (
            None
            if mr_metrics["localization_violation"] is None
            else float(mr_metrics["localization_violation"])
        ),
        "MR1_detection_preservation_violated": int(
            detection_binary
        ),
        "MR2_confidence_stability_violated": int(
            confidence_binary
        ),
        "MR3_localization_stability_violated": int(
            localization_binary
        ),
        "MR_score_positive": int(mr_score_binary),
        "F1_1_minus_mr_score": float(
            1.0 - mr_metrics["mr_score"]
        ),
        "F2_perturbation_magnitude": float(
            perturbation_magnitude
        ),
    }

    return (
        row,
        perturbed_image,
        perturbed_predictions,
    )


# Legacy visualization helper retained for optional/manual use only.
# The production final-output path below does not call save_candidate(),
# so candidate PNGs and per-candidate JSON files are not written.
def save_candidate(
    output_directory,
    prefix,
    x,
    orig_image,
    ref_preds,
    gt_boxes,
    model,
    class_names,
    device,
):
    (
        row,
        perturbed_image,
        perturbed_predictions,
    ) = evaluate_candidate(
        x=x,
        orig_image=orig_image,
        ref_preds=ref_preds,
        model=model,
        device=device,
    )

    cv2.imwrite(
        str(
            output_directory
            / f"{prefix}_perturbed.png"
        ),
        perturbed_image,
    )

    draw_predictions(
        image=perturbed_image,
        gt_boxes=gt_boxes,
        predictions=perturbed_predictions,
        class_names=class_names,
        output_path=(
            output_directory
            / f"{prefix}_gt_and_preds.png"
        ),
    )

    with open(
        output_directory / f"{prefix}_metrics.json",
        "w",
    ) as file:
        json.dump(
            json_safe(row),
            file,
            indent=2,
        )

    return row



# Per-image population-limited Random Search

def run_one_image(
    img_path,
    model,
    class_names,
    image_info_by_name,
    annotations_by_image_id,
    device,
    seed,
):
    """
    Run population-limited Random Search for one image.

    Fair comparison with the NSGA-II script:
      - Same perturbation variables and bounds.
      - Same population size.
      - Same number of generations.
      - Same nominal evaluation budget: POP_SIZE * N_GEN.
      - Same detector inference settings.
      - Same class-aware matching and MR calculations.
      - Same two objectives.
      - Same hypervolume reference point.
      - Same final output protocol.

    Processing:
      - Generation 1 samples POP_SIZE random candidates.
      - Each later generation samples POP_SIZE new random candidates.
      - The new candidates are merged with the previously retained
        population.
      - Exactly POP_SIZE solutions survive using non-dominated rank and
        crowding-distance filling.

    Hypervolume is calculated after every generation from the rank-0
    subset of the current retained population.

    At the end:
      - ALL_SAVED contains metadata for exactly the final retained POP_SIZE solutions.
      - The exact perturbation vectors are stored in saved_X.npy.
      - Candidate PNGs and per-candidate JSON files are not saved; they can be
        reconstructed from the original image and the stored perturbation vector.
      - Rank-0 solutions remain identifiable through their saved rank in ALL_SAVED.
    """
    image_output = OUT_ROOT / safe_name(img_path)

    all_saved_directory = image_output / "ALL_SAVED"

    all_saved_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)

    orig_image = cv2.imread(str(img_path))

    if orig_image is None:
        print(f"[skip] Could not load image: {img_path}")
        return None

    image_height, image_width = orig_image.shape[:2]

    gt_boxes = load_coco_gt_annotations(
        img_path=img_path,
        image_info_by_name=image_info_by_name,
        annotations_by_image_id=annotations_by_image_id,
        image_width=image_width,
        image_height=image_height,
    )

    ref_preds = run_torchvision_detector(
        model,
        orig_image,
        device,
    )

    print("\n" + "=" * 80)
    print(f"Image: {img_path.name}")
    print(f"GT objects: {len(gt_boxes)}")
    print(f"Reference predictions: {len(ref_preds)}")
    print("=" * 80)

    if len(ref_preds) == 0:
        print(
            f"[skip] No reference predictions for "
            f"{img_path.name}"
        )
        return None

    draw_predictions(
        image=orig_image,
        gt_boxes=gt_boxes,
        predictions=ref_preds,
        class_names=class_names,
        output_path=(
            image_output
            / "baseline_gt_and_preds.png"
        ),
    )

    convergence_rows = []
    first_violation_generation = None

    hv_indicator = HV(ref_point=HV_REF_POINT)
    hv_best_so_far = -np.inf
    hv_first_generation_at_best = None

    lower_bounds = np.array([
        BRIGHTNESS_MIN,
        CONTRAST_MIN,
        GAMMA_MIN,
        BLUR_MIN,
    ], dtype=float)

    upper_bounds = np.array([
        BRIGHTNESS_MAX,
        CONTRAST_MAX,
        GAMMA_MAX,
        BLUR_MAX,
    ], dtype=float)

    retained_X = None
    retained_F = None
    num_evaluations = 0

    for generation in range(1, N_GEN + 1):
        sampled_X = rng.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(POP_SIZE, 4),
        )

        sampled_F_rows = []

        for x in sampled_X:
            row, _, _ = evaluate_candidate(
                x=x,
                orig_image=orig_image,
                ref_preds=ref_preds,
                model=model,
                device=device,
            )

            sampled_F_rows.append([
                row["F1_1_minus_mr_score"],
                row["F2_perturbation_magnitude"],
            ])

        sampled_F = np.asarray(
            sampled_F_rows,
            dtype=float,
        )

        num_evaluations += len(sampled_X)

        if retained_X is None:
            candidate_X = sampled_X
            candidate_F = sampled_F
        else:
            candidate_X = np.vstack([
                retained_X,
                sampled_X,
            ])
            candidate_F = np.vstack([
                retained_F,
                sampled_F,
            ])

        (
            retained_X,
            retained_F,
            _,
        ) = select_rank_fill_with_crowding(
            F_pop=candidate_F,
            X_pop=candidate_X,
            n_select=POP_SIZE,
        )

        rank0_indices = get_rank0_indices(
            retained_F
        )

        if len(rank0_indices) == 0:
            continue

        pareto_F_current = retained_F[
            rank0_indices
        ]

        mr_scores = 1.0 - pareto_F_current[:, 0]
        magnitudes = pareto_F_current[:, 1]

        mr_min = float(np.min(mr_scores))
        mr_max = float(np.max(mr_scores))
        mr_mean = float(np.mean(mr_scores))
        mr_median = float(np.median(mr_scores))

        magnitude_min = float(np.min(magnitudes))
        magnitude_max = float(np.max(magnitudes))
        magnitude_mean = float(np.mean(magnitudes))
        magnitude_median = float(np.median(magnitudes))

        f1_min = float(np.min(pareto_F_current[:, 0]))
        f1_max = float(np.max(pareto_F_current[:, 0]))
        f2_min = float(np.min(pareto_F_current[:, 1]))
        f2_max = float(np.max(pareto_F_current[:, 1]))

        hv_value = compute_hv(
            pareto_F_current,
            hv_indicator,
        )

        hv_improvement = 0.0
        hv_relative_improvement = 0.0

        if hv_value > hv_best_so_far:
            if (
                np.isfinite(hv_best_so_far)
                and hv_best_so_far > 0
            ):
                hv_improvement = (
                    hv_value - hv_best_so_far
                )
                hv_relative_improvement = (
                    hv_improvement / hv_best_so_far
                )
            elif not np.isfinite(hv_best_so_far):
                hv_improvement = hv_value

            hv_best_so_far = hv_value
            hv_first_generation_at_best = generation

        if (
            first_violation_generation is None
            and mr_max > 0
        ):
            first_violation_generation = generation

        convergence_rows.append({
            "generation": int(generation),
            "n_eval": int(num_evaluations),
            "retained_population_size": int(
                len(retained_F)
            ),
            "pareto_size": int(
                len(pareto_F_current)
            ),
            "hv": float(hv_value),
            "hv_best_so_far": float(
                hv_best_so_far
            ),
            "hv_improvement": float(
                hv_improvement
            ),
            "hv_relative_improvement": float(
                hv_relative_improvement
            ),
            "hv_best_generation_so_far": (
                hv_first_generation_at_best
                if hv_first_generation_at_best is not None
                else ""
            ),
            "mr_min": mr_min,
            "mr_max": mr_max,
            "mr_mean": mr_mean,
            "mr_median": mr_median,
            "mag_min": magnitude_min,
            "mag_max": magnitude_max,
            "mag_mean": magnitude_mean,
            "mag_median": magnitude_median,
            "f1_min": f1_min,
            "f1_max": f1_max,
            "f2_min": f2_min,
            "f2_max": f2_max,
            "first_violation_generation_so_far": (
                first_violation_generation
                if first_violation_generation is not None
                else ""
            ),
        })

        if generation % 25 == 0 or generation == 1:
            print(
                f"Gen {generation:04d} | "
                f"evals={num_evaluations} | "
                f"retained={len(retained_F)} | "
                f"Pareto n={len(pareto_F_current)} | "
                f"HV={hv_value:.6f} | "
                f"HV*={hv_best_so_far:.6f} | "
                f"MR range=[{mr_min:.4f}, {mr_max:.4f}] | "
                f"Mag range=[{magnitude_min:.4f}, "
                f"{magnitude_max:.4f}]"
            )

    save_csv(
        image_output / "convergence.csv",
        convergence_rows,
    )

    if retained_X is None or retained_F is None:
        raise RuntimeError(
            "Random Search completed without a final population."
        )

    if (
        len(retained_X) != POP_SIZE
        or len(retained_F) != POP_SIZE
    ):
        print(
            f"[warn] Expected {POP_SIZE} final solutions, "
            f"but obtained X={len(retained_X)}, "
            f"F={len(retained_F)}."
        )

    final_ranks = get_all_ranks(retained_F)

    final_rank0_indices = np.where(
        final_ranks == 0
    )[0]

    pareto_X = retained_X[final_rank0_indices]
    pareto_F = retained_F[final_rank0_indices]

    # ========================================================
    # Save ALL_SAVED:
    # exactly the final retained Random Search population
    # ========================================================
    # Keep the exact optimizer values so every perturbation can be
    # reconstructed later without precision loss.
    np.save(
        all_saved_directory / "saved_X.npy",
        retained_X,
    )
    np.save(
        all_saved_directory / "saved_F.npy",
        retained_F,
    )

    saved_rows = []

    for candidate_index, (
        x,
        rank,
    ) in enumerate(
        zip(retained_X, final_ranks),
        start=1,
    ):
        # Re-evaluate each final retained solution once to collect all scalar
        # MR/reporting metrics. No candidate image or per-candidate JSON is
        # written here; saved_X.npy + the original image are sufficient to
        # reconstruct the perturbation later.
        row, _, _ = evaluate_candidate(
            x=x,
            orig_image=orig_image,
            ref_preds=ref_preds,
            model=model,
            device=device,
        )

        row = {
            "image_name": img_path.name,
            "candidate_id": int(candidate_index),
            "rank": int(rank),
            **row,
        }

        saved_rows.append(row)

    save_csv(
        all_saved_directory
        / "saved_candidates.csv",
        saved_rows,
    )

    with open(
        all_saved_directory / "saved_meta.json",
        "w",
    ) as file:
        json.dump(
            json_safe({
                "image": img_path.name,
                "method": "Random Search",
                "model_type": MODEL_NAME,
                "seed": int(seed),
                "saved_n": int(len(retained_X)),
                "target_n": int(POP_SIZE),
                "nominal_total_evaluations": int(
                    POP_SIZE * N_GEN
                ),
                "actual_total_evaluations": int(
                    num_evaluations
                ),
                "selection": (
                    "Final retained population after repeated "
                    "random sampling, non-dominated sorting, and "
                    "crowding-distance filling; rank-0 solutions "
                    "are identified by their saved rank."
                ),
                "candidate_storage": (
                    "Metadata only: candidate PNGs and per-candidate JSON files "
                    "are not saved. Reconstruct a candidate by loading the "
                    "original image and applying its decision vector from "
                    "saved_X.npy with apply_illumination_and_blur()."
                ),
                "saved_candidate_files": [
                    "saved_X.npy",
                    "saved_F.npy",
                    "saved_candidates.csv",
                    "saved_meta.json",
                ],
                "rank_counts": {
                    str(int(rank)): int(
                        np.sum(final_ranks == rank)
                    )
                    for rank in np.unique(final_ranks)
                },
            }),
            file,
            indent=2,
        )

    best_saved = (
        max(
            saved_rows,
            key=lambda row: row["mr_score"],
        )
        if saved_rows
        else None
    )

    summary = {
        "method": "Random Search",
        "model_type": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "image_name": img_path.name,
        "seed": int(seed),
        "population": int(POP_SIZE),
        "generations": int(N_GEN),
        "nominal_total_evaluations": int(
            POP_SIZE * N_GEN
        ),
        "actual_total_evaluations": int(
            num_evaluations
        ),
        "score_threshold": float(
            SCORE_THRESHOLD
        ),
        "nms_threshold": float(
            NMS_THRESHOLD
        ),
        "pred_match_iou_threshold": float(
            PRED_MATCH_IOU_THRESHOLD
        ),
        "num_gt_objects": int(len(gt_boxes)),
        "num_reference_predictions": int(
            len(ref_preds)
        ),
        "first_violation_generation": (
            first_violation_generation
        ),
        "num_pareto_solutions": int(
            len(pareto_X)
        ),
        "num_saved_solutions": int(
            len(saved_rows)
        ),
        "best_saved_mr_score": (
            None
            if best_saved is None
            else best_saved["mr_score"]
        ),
        "best_saved_magnitude": (
            None
            if best_saved is None
            else best_saved[
                "perturbation_magnitude"
            ]
        ),
        "final_hv": (
            None
            if not convergence_rows
            else convergence_rows[-1]["hv"]
        ),
        "best_hv": (
            None
            if not convergence_rows
            else max(
                row["hv_best_so_far"]
                for row in convergence_rows
            )
        ),
        "best_hv_generation": (
            None
            if not convergence_rows
            else convergence_rows[-1][
                "hv_best_generation_so_far"
            ]
        ),
        "num_saved_detection_violations": sum(
            row[
                "MR1_detection_preservation_violated"
            ]
            for row in saved_rows
        ),
        "num_saved_confidence_violations": sum(
            row[
                "MR2_confidence_stability_violated"
            ]
            for row in saved_rows
        ),
        "num_saved_localization_violations": sum(
            row[
                "MR3_localization_stability_violated"
            ]
            for row in saved_rows
        ),
        "num_saved_mr_score_positive": sum(
            row["MR_score_positive"]
            for row in saved_rows
        ),
        "total_lost_predictions_saved": sum(
            row["lost_predictions"]
            for row in saved_rows
        ),
        "total_new_predictions_saved": sum(
            row["new_predictions"]
            for row in saved_rows
        ),
    }

    with open(
        image_output / "summary.json",
        "w",
    ) as file:
        json.dump(
            json_safe(summary),
            file,
            indent=2,
        )

    print(
        f"[done] {img_path.name} | "
        f"best saved MR="
        f"{summary['best_saved_mr_score']} | "
        f"first violation gen="
        f"{first_violation_generation}"
    )

    return summary



# Main

def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_DIR}"
        )

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: "
            f"{ANNOTATIONS_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    (
        image_info_by_name,
        annotations_by_image_id,
        class_names,
        num_classes,
    ) = load_coco_dataset(
        ANNOTATIONS_PATH
    )

    device = resolve_device()

    model = load_torchvision_model(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        num_classes=num_classes,
        device=device,
    )

    image_files = sorted([
        path
        for path in IMAGES_DIR.iterdir()
        if path.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    ])

    print(f"Images directory: {IMAGES_DIR}")
    print(f"Annotations: {ANNOTATIONS_PATH}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Model checkpoint: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Images found: {len(image_files)}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Classes: {class_names}")

    all_summaries = []

    for image_index, img_path in enumerate(
        image_files
    ):
        seed = BASE_SEED + image_index

        image_output = (
            OUT_ROOT / safe_name(img_path)
        )
        summary_path = (
            image_output / "summary.json"
        )

        # Resume support:
        # skip an image only if its summary belongs to the
        # current experiment configuration.
        if summary_path.exists():
            try:
                with open(summary_path, "r") as file:
                    existing_summary = json.load(file)

                summary_matches = (
                    existing_summary.get("image_name")
                    == img_path.name
                    and existing_summary.get("method")
                    == "Random Search"
                    and existing_summary.get("model_path")
                    == str(MODEL_PATH)
                    and existing_summary.get("model_type")
                    == MODEL_NAME
                    and existing_summary.get("population")
                    == POP_SIZE
                    and existing_summary.get("generations")
                    == N_GEN
                    and existing_summary.get(
                        "score_threshold"
                    )
                    == SCORE_THRESHOLD
                    and existing_summary.get(
                        "nms_threshold"
                    )
                    == NMS_THRESHOLD
                    and existing_summary.get(
                        "pred_match_iou_threshold"
                    )
                    == PRED_MATCH_IOU_THRESHOLD
                )

                if summary_matches:
                    print(
                        f"[resume] Skipping completed "
                        f"image: {img_path.name}"
                    )
                    all_summaries.append(
                        existing_summary
                    )
                    continue

                print(
                    f"[warn] Existing summary does not "
                    f"match the current configuration for "
                    f"{img_path.name}. Reprocessing."
                )

            except Exception as error:
                print(
                    f"[warn] Could not read existing "
                    f"summary for {img_path.name}: "
                    f"{error}. Reprocessing."
                )

        try:
            summary = run_one_image(
                img_path=img_path,
                model=model,
                class_names=class_names,
                image_info_by_name=image_info_by_name,
                annotations_by_image_id=annotations_by_image_id,
                device=device,
                seed=seed,
            )

            if summary is not None:
                all_summaries.append(summary)

        except Exception as error:
            print(
                f"[ERROR] image={img_path.name}: "
                f"{error}"
            )

    if all_summaries:
        save_csv(
            OUT_ROOT / "summary_all_images.csv",
            all_summaries,
        )

    print("\n=== All done ===")
    print(
        f"Saved summary to: "
        f"{OUT_ROOT / 'summary_all_images.csv'}"
    )


if __name__ == "__main__":
    main()


