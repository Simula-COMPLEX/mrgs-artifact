import cv2
import csv
import json
import random
import numpy as np
from pathlib import Path
import os

from ultralytics import YOLO
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV


# Paths

PROJECT_ROOT = Path.cwd()

IMAGES_DIR = PROJECT_ROOT / "all_data/images"
LABELS_DIR = PROJECT_ROOT / "all_data/labels"
MODEL_PATH = PROJECT_ROOT / "model/best.pt"

# Created only to group all selected images into a single directory
CATEGORY = "all_data"

# Keep the output separate from the dataset and from the NSGA-II results.
OUT_ROOT = PROJECT_ROOT / "nsga2_jaccard_noscrews" / CATEGORY
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_ID = os.environ.get("RUN_ID", "")
if RUN_ID:
    OUT_ROOT = OUT_ROOT / RUN_ID

OUT_ROOT.mkdir(parents=True, exist_ok=True)

POP_SIZE = 40
N_GEN = 250
BASE_SEED = int(os.environ.get("BASE_SEED", 42))

# YOLO inference settings.
#
# These values can be changed if the trained model was evaluated with
# different thresholds. They are applied identically to the original and
# perturbed images.
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.50
TOP_K = 5000
YOLO_IMGSZ = 640

# None lets Ultralytics select the available device automatically.
# On a SLURM GPU allocation, this will normally use the visible GPU.
# Examples of explicit values: 0, "cuda:0", or "cpu".
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
    Get a readable class name from Ultralytics model.names.
    """
    if isinstance(class_names, dict):
        return str(class_names.get(class_id, class_id))

    if isinstance(class_names, (list, tuple)):
        if 0 <= class_id < len(class_names):
            return str(class_names[class_id])

    return str(class_id)



# YOLO ground-truth loading
#
# Ground truth is used only for visualization and metadata.
# It is not used in the MR oracle, search objectives, selection,
# or hypervolume calculation.

def load_yolo_gt_annotations(
    label_path: Path,
    image_width: int,
    image_height: int,
):
    """
    Load one YOLO-format annotation file.

    Expected line format:
        class_id x_center y_center width height

    The four box coordinates are normalized to [0, 1].
    """
    boxes = []

    if not label_path.exists():
        print(f"[warn] Missing label file: {label_path}")
        return boxes

    with open(label_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"[warn] Invalid label line "
                    f"{label_path}:{line_number}: {line}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
            except ValueError:
                print(
                    f"[warn] Could not parse label line "
                    f"{label_path}:{line_number}: {line}"
                )
                continue

            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            box_width = width_norm * image_width
            box_height = height_norm * image_height

            x1 = x_center - box_width / 2.0
            y1 = y_center - box_height / 2.0
            x2 = x_center + box_width / 2.0
            y2 = y_center + box_height / 2.0

            x1 = float(np.clip(x1, 0.0, max(image_width - 1, 0)))
            y1 = float(np.clip(y1, 0.0, max(image_height - 1, 0)))
            x2 = float(np.clip(x2, 0.0, max(image_width - 1, 0)))
            y2 = float(np.clip(y2, 0.0, max(image_height - 1, 0)))

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



# YOLO inference

def run_yolo(model, image):
    """
    Run Ultralytics YOLO inference and return predictions in the
    common structure used by the MR implementation.
    """
    predict_kwargs = {
        "source": image,
        "conf": SCORE_THRESHOLD,
        "iou": NMS_THRESHOLD,
        "max_det": TOP_K,
        "imgsz": YOLO_IMGSZ,
        "verbose": False,
        "save": False,
        "agnostic_nms": False,
    }

    if DEVICE is not None:
        predict_kwargs["device"] = DEVICE

    results = model.predict(**predict_kwargs)

    predictions = []

    if not results:
        return predictions

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return predictions

    xyxy_boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, confidence, class_id in zip(
        xyxy_boxes,
        confidences,
        class_ids,
    ):
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
    Apply the same global perturbation sequence used in the face case:

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
        YOLO predictions on the original image.

    Follow-up:
        YOLO predictions on the perturbed image.

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

            # Preserve the directional confidence logic of the face case:
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
):
    perturbed_image = apply_illumination_and_blur(
        orig_image,
        x,
    )

    perturbed_predictions = run_yolo(
        model,
        perturbed_image,
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



# Per-image NSGA-II

def run_one_image(
    img_path,
    model,
    class_names,
    seed,
):
    """
    Run NSGA-II for one image.

    Fair comparison with the population-limited random-search script:
      - Same perturbation variables and bounds.
      - Same population size.
      - Same number of generations.
      - Same nominal evaluation budget: POP_SIZE * N_GEN.
      - Same YOLO inference settings.
      - Same class-aware matching and MR calculations.
      - Same two objectives.
      - Same hypervolume reference point.
      - Same final output protocol.

    Hypervolume is calculated after every generation from the rank-0
    subset of the current retained NSGA-II population of POP_SIZE
    solutions.

    At the end:
      - ALL_SAVED contains exactly the final retained POP_SIZE solutions.
      - Rank-0 solutions remain identifiable through the saved rank values.
      - No duplicate ALL_PARETO directory or per-candidate images are saved.
    """
    image_output = OUT_ROOT / safe_name(img_path)

    all_saved_directory = image_output / "ALL_SAVED"

    all_saved_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(seed)
    np.random.seed(seed)

    orig_image = cv2.imread(str(img_path))

    if orig_image is None:
        print(f"[skip] Could not load image: {img_path}")
        return None

    image_height, image_width = orig_image.shape[:2]

    label_path = LABELS_DIR / f"{img_path.stem}.txt"

    gt_boxes = load_yolo_gt_annotations(
        label_path=label_path,
        image_width=image_width,
        image_height=image_height,
    )

    ref_preds = run_yolo(
        model,
        orig_image,
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

    class MRGuidedYoloProblem(ElementwiseProblem):
        def __init__(self):
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

            super().__init__(
                n_var=4,
                n_obj=2,
                n_constr=0,
                xl=lower_bounds,
                xu=upper_bounds,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            row, _, _ = evaluate_candidate(
                x=x,
                orig_image=orig_image,
                ref_preds=ref_preds,
                model=model,
            )

            out["F"] = [
                row["F1_1_minus_mr_score"],
                row["F2_perturbation_magnitude"],
            ]

    def callback(algorithm):
        nonlocal first_violation_generation
        nonlocal hv_best_so_far
        nonlocal hv_first_generation_at_best

        generation = int(algorithm.n_gen)
        num_evaluations = int(
            getattr(algorithm.evaluator, "n_eval", 0)
        )

        # Use the complete current retained population. Extracting its
        # rank-0 subset makes the HV protocol explicit and directly
        # comparable with the population-limited random-search script.
        retained_F_current = np.asarray(
            algorithm.pop.get("F"),
            dtype=float,
        )

        if (
            retained_F_current.ndim != 2
            or len(retained_F_current) == 0
        ):
            return

        rank0_indices = get_rank0_indices(
            retained_F_current
        )

        if len(rank0_indices) == 0:
            return

        pareto_F_current = retained_F_current[
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
                len(retained_F_current)
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
                f"retained={len(retained_F_current)} | "
                f"Pareto n={len(pareto_F_current)} | "
                f"HV={hv_value:.6f} | "
                f"HV*={hv_best_so_far:.6f} | "
                f"MR range=[{mr_min:.4f}, {mr_max:.4f}] | "
                f"Mag range=[{magnitude_min:.4f}, "
                f"{magnitude_max:.4f}]"
            )

    problem = MRGuidedYoloProblem()

    algorithm = NSGA2(
        pop_size=POP_SIZE,
    )

    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=("n_gen", N_GEN),
        seed=seed,
        verbose=False,
        callback=callback,
    )

    save_csv(
        image_output / "convergence.csv",
        convergence_rows,
    )

    if result.pop is None:
        raise RuntimeError(
            "NSGA-II completed without a final population."
        )

    # The final retained NSGA-II population contains POP_SIZE solutions.
    retained_X = np.asarray(
        result.pop.get("X"),
        dtype=float,
    )
    retained_F = np.asarray(
        result.pop.get("F"),
        dtype=float,
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
    # exactly the final retained NSGA-II population
    # ========================================================
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
        # Re-evaluate only to recover the scalar candidate metrics for CSV.
        # Perturbed images, prediction visualizations, and per-candidate JSON
        # files are intentionally not saved in the compact output format.
        row, _, _ = evaluate_candidate(
            x=x,
            orig_image=orig_image,
            ref_preds=ref_preds,
            model=model,
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
                "method": "NSGA-II",
                "model_type": "Ultralytics YOLO",
                "seed": int(seed),
                "saved_n": int(len(retained_X)),
                "target_n": int(POP_SIZE),
                "nominal_total_evaluations": int(
                    POP_SIZE * N_GEN
                ),
                "actual_total_evaluations": (
                    None
                    if not convergence_rows
                    else convergence_rows[-1]["n_eval"]
                ),
                "selection": (
                    "Final retained NSGA-II population; rank-0 "
                    "solutions are identified by rank == 0 in "
                    "saved_candidates.csv."
                ),
                "storage_policy": (
                    "Compact reproducible output: saved_X.npy and "
                    "saved_F.npy preserve the final optimizer population; "
                    "per-candidate perturbed images, visualizations, and "
                    "JSON files are not stored."
                ),
                "candidate_artifacts_reconstructable": True,
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
        "method": "NSGA-II",
        "model_type": "Ultralytics YOLO",
        "model_path": str(MODEL_PATH),
        "image_name": img_path.name,
        "seed": int(seed),
        "population": int(POP_SIZE),
        "generations": int(N_GEN),
        "nominal_total_evaluations": int(
            POP_SIZE * N_GEN
        ),
        "actual_total_evaluations": (
            None
            if not convergence_rows
            else convergence_rows[-1]["n_eval"]
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

    if not LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Labels directory not found: {LABELS_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))
    class_names = model.names

    image_files = sorted([
        path
        for path in IMAGES_DIR.iterdir()
        if path.suffix.lower()
        in [".jpg", ".jpeg", ".png"]
    ])

    print(f"Images directory: {IMAGES_DIR}")
    print(f"Labels directory: {LABELS_DIR}")
    print(f"Model: {MODEL_PATH}")
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
                    == "NSGA-II"
                    and existing_summary.get("model_path")
                    == str(MODEL_PATH)
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
