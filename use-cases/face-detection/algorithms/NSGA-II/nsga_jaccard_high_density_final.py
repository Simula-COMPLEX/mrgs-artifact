import os
import cv2
import csv
import json
import random
import numpy as np
from pathlib import Path

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
cv2.setNumThreads(N_THREADS)

PROJECT_ROOT = Path.cwd()

DATASET_ROOT = PROJECT_ROOT / "WIDER_val_density_600"
#MODEL_PATH = PROJECT_ROOT / "models/face_detection_yunet_2026may.onnx"
MODEL_PATH = PROJECT_ROOT / "models/face_detection_yunet_2023mar.onnx"


# CATEGORY = "few_faces"
CATEGORY = "high_density"
FINAL_CATEGORY = "high_density_final"

IMAGE_DIR = DATASET_ROOT / "images" / CATEGORY
GT_FILE = DATASET_ROOT / "annotations" / "sampled_600_valid_only_gt.txt"

OUT_ROOT = DATASET_ROOT / "nsga_yunet_jaccard_2023_high" / FINAL_CATEGORY
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUN_ID = os.environ.get("RUN_ID", "")
if RUN_ID:
    OUT_ROOT = OUT_ROOT / RUN_ID

OUT_ROOT.mkdir(parents=True, exist_ok=True)

POP_SIZE = 40
N_GEN = 250
BASE_SEED = int(os.environ.get("BASE_SEED", 42))

DEVICE = os.environ.get("DEVICE", None)

SCORE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Used for matching original detections to perturbed detections.
PRED_MATCH_IOU_THRESHOLD = 0.5

# Search vector:
# x = [brightness_delta, contrast_factor, gamma_value, blur_strength]
BRIGHTNESS_MIN, BRIGHTNESS_MAX = -40.0, 40.0
CONTRAST_MIN, CONTRAST_MAX = 0.6, 1.6
GAMMA_MIN, GAMMA_MAX = 0.5, 2.0
BLUR_MIN, BLUR_MAX = 0.0, 15.0

OBJECTIVE_NAMES = [
    "one_minus_mr_score",
    "perturbation_magnitude"
]

# Hypervolume configuration. Both objectives are in [0, 1]:
#   F1 = 1 - MR_score
#   F2 = perturbation_magnitude
# The reference point is intentionally slightly worse than the worst possible value.
HV_REF_POINT = np.array([1.1, 1.1], dtype=float)

def safe_name(path: Path) -> str:
    return path.stem.replace("/", "_").replace(" ", "_")


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box_a, box_b) -> float:
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def save_csv(path: Path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def json_safe(obj):
    """Convert numpy scalars and NaNs to JSON-friendly values."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj



# GT loading, used only for visualization/debugging

def load_gt_annotations(gt_file: Path, target_category: str):
    annotations = {}

    with open(gt_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0

    while i < len(lines):
        rel_img_path = lines[i]
        i += 1

        num_faces = int(lines[i])
        i += 1

        category = Path(rel_img_path).parent.name
        image_name = Path(rel_img_path).name

        boxes = []

        for _ in range(num_faces):
            parts = list(map(int, lines[i].split()))
            i += 1

            x, y, w, h = parts[:4]
            blur, expression, illumination, invalid, occlusion, pose = parts[4:10]

            if category == target_category:
                boxes.append({
                    "bbox_xywh": [x, y, w, h],
                    "bbox_xyxy": xywh_to_xyxy([x, y, w, h]),
                    "blur": blur,
                    "expression": expression,
                    "illumination": illumination,
                    "invalid": invalid,
                    "occlusion": occlusion,
                    "pose": pose
                })

        if category == target_category:
            annotations[image_name] = boxes

    return annotations



# YuNet inference

def extract_yunet_predictions(faces):
    preds = []

    if faces is None:
        return preds

    for face in faces:
        x, y, w, h = face[:4]
        score = float(face[14])

        preds.append({
            "bbox_xywh": [float(x), float(y), float(w), float(h)],
            "bbox_xyxy": xywh_to_xyxy([float(x), float(y), float(w), float(h)]),
            "confidence": score
        })

    return preds


def run_yunet(detector, img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    return extract_yunet_predictions(faces)



# One-to-one matching for MR oracle

def match_predictions_to_reference(ref_preds, pert_preds, iou_threshold=0.5):
    """
    One-to-one IoU-based matching between original/reference predictions and
    perturbed predictions.

    A reference detection is preserved only if the best unused perturbed
    detection has IoU >= iou_threshold.

    Returns:
      matches: list of reference-side records
      used_pert: set of perturbed indices matched to a reference prediction
    """
    matches = []
    used_pert = set()

    for ref_idx, ref in enumerate(ref_preds):
        best_iou = 0.0
        best_pert_idx = None

        for pert_idx, pert in enumerate(pert_preds):
            if pert_idx in used_pert:
                continue

            iou = compute_iou(ref["bbox_xyxy"], pert["bbox_xyxy"])

            if iou > best_iou:
                best_iou = iou
                best_pert_idx = pert_idx

        if best_pert_idx is not None and best_iou >= iou_threshold:
            used_pert.add(best_pert_idx)

            matches.append({
                "ref_idx": ref_idx,
                "pert_idx": best_pert_idx,
                "matched": True,
                "iou": float(best_iou),
                "ref_confidence": float(ref["confidence"]),
                "pert_confidence": float(pert_preds[best_pert_idx]["confidence"])
            })
        else:
            matches.append({
                "ref_idx": ref_idx,
                "pert_idx": None,
                "matched": False,
                "iou": 0.0,
                "ref_confidence": float(ref["confidence"]),
                "pert_confidence": 0.0
            })

    return matches, used_pert



# Perturbation operators

def apply_illumination_and_blur(img, x):
    brightness_delta, contrast_factor, gamma_value, blur_strength = x

    img_f = img.astype(np.float32)

    # Brightness + contrast
    img_f = img_f * contrast_factor + brightness_delta
    img_f = np.clip(img_f, 0, 255)

    # Gamma correction
    img_norm = img_f / 255.0
    img_gamma = np.power(img_norm, gamma_value) * 255.0
    img_out = np.clip(img_gamma, 0, 255).astype(np.uint8)

    # Gaussian blur
    if blur_strength >= 1.0:
        k = int(round(blur_strength))
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        img_out = cv2.GaussianBlur(img_out, (k, k), 0)

    return img_out


def perturbation_magnitude_image_space(orig_img, pert_img) -> float:
    """
    Generic image-space perturbation magnitude:
    normalized 95th percentile absolute pixel difference.
    """
    diff = np.abs(
        orig_img.astype(np.float32) -
        pert_img.astype(np.float32)
    )

    magnitude = np.percentile(diff, 95) / 255.0

    return float(np.clip(magnitude, 0.0, 1.0))



# MR metrics

def compute_mr_metrics_from_predictions(ref_preds, pert_preds):
    """
    Label-free MR score.

    Reference = YuNet predictions on original image.
    Follow-up = YuNet predictions on perturbed image.

    MR1: Detection Preservation
      - Uses Jaccard distance over detection sets after one-to-one IoU matching.
      - matched = preserved detections
      - lost    = reference detections not matched
      - new     = perturbed detections not matched to any reference detection
      - DetectionViolation = 1 - matched / (matched + lost + new)

    MR2: Confidence Stability
      - Computed only over preserved detections.
      - If no detections are preserved, this component is undefined.

    MR3: Localization Stability
      - Computed only over preserved detections.
      - If no detections are preserved, this component is undefined.

    Final MR_score:
      - If at least one detection is preserved: mean(MR1, MR2, MR3).
      - If none are preserved: MR_score = MR1.
    """
    n_ref = len(ref_preds)
    n_pert = len(pert_preds)

    if n_ref == 0:
        # No reference behavior to preserve. Skip such images in the main loop,
        # but keep this fallback to avoid crashes.
        return {
            "num_reference_predictions": 0,
            "num_perturbed_predictions": n_pert,
            "matched_after": 0,
            "lost_predictions": 0,
            "new_predictions": n_pert,
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
        ref_preds,
        pert_preds,
        iou_threshold=PRED_MATCH_IOU_THRESHOLD
    )

    matched = [m for m in matches if m["matched"]]
    matched_after = len(matched)

    lost_predictions = n_ref - matched_after
    new_predictions = n_pert - len(used_pert)

    union_size = matched_after + lost_predictions + new_predictions
    if union_size == 0:
        detection_jaccard_similarity = 1.0
    else:
        detection_jaccard_similarity = matched_after / union_size

    detection_violation = 1.0 - detection_jaccard_similarity

    # Keep detected_ratio for interpretability. This captures preservation only,
    # while detection_violation captures both lost and new detections.
    detected_ratio = matched_after / n_ref

    if matched_after == 0:
        confidence_ratio = None
        mean_iou = None
        confidence_violation = None
        localization_violation = None

        # Only detection-set violation is defined if no detection survives.
        mr_score = detection_violation

    else:
        confidence_ratios = []
        ious = []

        for m in matched:
            conf_ratio = m["pert_confidence"] / max(m["ref_confidence"], 1e-9)
            conf_ratio = float(np.clip(conf_ratio, 0.0, 1.0))

            confidence_ratios.append(conf_ratio)
            ious.append(float(np.clip(m["iou"], 0.0, 1.0)))

        confidence_ratio = float(np.mean(confidence_ratios))
        mean_iou = float(np.mean(ious))

        confidence_violation = 1.0 - confidence_ratio
        localization_violation = 1.0 - mean_iou

        mr_score = float(np.mean([
            detection_violation,
            confidence_violation,
            localization_violation
        ]))

    return {
        "num_reference_predictions": int(n_ref),
        "num_perturbed_predictions": int(n_pert),
        "matched_after": int(matched_after),
        "lost_predictions": int(lost_predictions),
        "new_predictions": int(new_predictions),
        "detection_jaccard_similarity": float(detection_jaccard_similarity),
        "detected_ratio": float(detected_ratio),
        "confidence_ratio": confidence_ratio,
        "mean_iou": mean_iou,
        "detection_violation": float(detection_violation),
        "confidence_violation": confidence_violation,
        "localization_violation": localization_violation,
        "mr_score": float(mr_score),
    }



# Pareto-rank / HV utilities

def get_rank0_front(F):
    """Return the rank-0 non-dominated objective vectors from F."""
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return F.reshape(0, len(OBJECTIVE_NAMES))

    nds = NonDominatedSorting()
    fronts = nds.do(F, only_non_dominated_front=False)

    if not fronts:
        return F.reshape(0, F.shape[1])

    rank0 = np.asarray(fronts[0], dtype=int)
    return F[rank0]


def get_all_ranks(F):
    """Return the non-dominated rank assigned to every objective vector."""
    F = np.asarray(F, dtype=float)

    if F.size == 0:
        return np.array([], dtype=int)

    nds = NonDominatedSorting()
    _, ranks = nds.do(F, return_rank=True)

    return np.asarray(ranks, dtype=int)


def compute_hv(F, hv_indicator):
    """Compute HV safely from a set of objective vectors."""
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0

    # Values should already be within [0, 1], but clipping avoids numerical issues
    # if a future objective produces tiny out-of-range floating-point values.
    F = np.clip(F, 0.0, HV_REF_POINT)
    return float(hv_indicator(F))


# Visualization / candidate evaluation

def draw_predictions(img, gt_boxes, preds, out_path):
    vis = img.copy()

    # GT in blue, only for visual debugging/evaluation.
    for gt in gt_boxes:
        x1, y1, x2, y2 = map(int, gt["bbox_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Predictions in green.
    for pred in preds:
        x1, y1, x2, y2 = map(int, pred["bbox_xyxy"])
        conf = pred["confidence"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{conf:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(str(out_path), vis)


def evaluate_candidate(x, orig_img, ref_preds, gt_boxes, detector):
    pert_img = apply_illumination_and_blur(orig_img, x)
    pert_preds = run_yunet(detector, pert_img)

    mr = compute_mr_metrics_from_predictions(ref_preds, pert_preds)
    magnitude = perturbation_magnitude_image_space(orig_img, pert_img)

    detection_binary = mr["detection_violation"] > 0

    confidence_binary = (
        mr["confidence_violation"] is not None and
        mr["confidence_violation"] > 0
    )

    localization_binary = (
        mr["localization_violation"] is not None and
        mr["localization_violation"] > 0
    )

    mr_score_binary = mr["mr_score"] > 0

    brightness_delta, contrast_factor, gamma_value, blur_strength = x

    row = {
        "brightness_delta": float(brightness_delta),
        "contrast_factor": float(contrast_factor),
        "gamma_value": float(gamma_value),
        "blur_strength": float(blur_strength),
        "perturbation_magnitude": float(magnitude),
        "mr_score": float(mr["mr_score"]),
        "num_reference_predictions": int(mr["num_reference_predictions"]),
        "num_perturbed_predictions": int(mr["num_perturbed_predictions"]),
        "matched_after": int(mr["matched_after"]),
        "lost_predictions": int(mr["lost_predictions"]),
        "new_predictions": int(mr["new_predictions"]),
        "detection_jaccard_similarity": float(mr["detection_jaccard_similarity"]),
        "detected_ratio": float(mr["detected_ratio"]),
        "confidence_ratio": None if mr["confidence_ratio"] is None else float(mr["confidence_ratio"]),
        "mean_iou": None if mr["mean_iou"] is None else float(mr["mean_iou"]),
        "detection_violation": float(mr["detection_violation"]),
        "confidence_violation": None if mr["confidence_violation"] is None else float(mr["confidence_violation"]),
        "localization_violation": None if mr["localization_violation"] is None else float(mr["localization_violation"]),
        "MR1_detection_preservation_violated": int(detection_binary),
        "MR2_confidence_stability_violated": int(confidence_binary),
        "MR3_localization_stability_violated": int(localization_binary),
        "MR_score_positive": int(mr_score_binary),
        "F1_1_minus_mr_score": float(1.0 - mr["mr_score"]),
        "F2_perturbation_magnitude": float(magnitude),
    }

    return row, pert_img, pert_preds



# Per-image NSGA-II

def run_one_image(img_path, gt_boxes, detector, seed):
    image_out = OUT_ROOT / safe_name(img_path)

    all_saved_dir = image_out / "ALL_SAVED"
    all_saved_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    orig_img = cv2.imread(str(img_path))

    if orig_img is None:
        print(f"[skip] Could not load image: {img_path}")
        return None

    ref_preds = run_yunet(detector, orig_img)

    print("\n" + "=" * 80)
    print(f"Image: {img_path.name}")
    print(f"GT faces: {len(gt_boxes)}")
    print(f"Reference predictions: {len(ref_preds)}")
    print("=" * 80)

    if len(ref_preds) == 0:
        print(f"[skip] No reference predictions for {img_path.name}")
        return None

    draw_predictions(
        orig_img,
        gt_boxes,
        ref_preds,
        image_out / "baseline_gt_and_preds.png"
    )

    conv_rows = []
    first_violation_gen = None

    hv_indicator = HV(ref_point=HV_REF_POINT)
    hv_best_so_far = -np.inf
    hv_first_gen_at_best = None

    class MRGuidedFaceProblem(ElementwiseProblem):
        def __init__(self):
            xl = np.array([
                BRIGHTNESS_MIN,
                CONTRAST_MIN,
                GAMMA_MIN,
                BLUR_MIN
            ], dtype=float)

            xu = np.array([
                BRIGHTNESS_MAX,
                CONTRAST_MAX,
                GAMMA_MAX,
                BLUR_MAX
            ], dtype=float)

            super().__init__(
                n_var=4,
                n_obj=2,
                n_constr=0,
                xl=xl,
                xu=xu
            )

        def _evaluate(self, x, out, *args, **kwargs):
            row, _, _ = evaluate_candidate(
                x=x,
                orig_img=orig_img,
                ref_preds=ref_preds,
                gt_boxes=gt_boxes,
                detector=detector
            )

            out["F"] = [
                row["F1_1_minus_mr_score"],
                row["F2_perturbation_magnitude"]
            ]

    def callback(algorithm):
        nonlocal first_violation_gen, hv_best_so_far, hv_first_gen_at_best

        gen = int(algorithm.n_gen)
        n_eval = int(getattr(algorithm.evaluator, "n_eval", 0))

        # Use the current rank-0 front for convergence analysis.
        # algorithm.opt is usually rank-0, but we fall back to pop and explicitly
        # extract rank-0 to keep this robust across pymoo versions/algorithms.
        if getattr(algorithm, "opt", None) is not None:
            F_current = np.asarray(algorithm.opt.get("F"), dtype=float)
        else:
            F_current = np.asarray(algorithm.pop.get("F"), dtype=float)

        if F_current is None or len(F_current) == 0:
            return

        F_front = get_rank0_front(F_current)
        if F_front is None or len(F_front) == 0:
            return

        mr_scores = 1.0 - F_front[:, 0]
        magnitudes = F_front[:, 1]

        mr_min = float(np.min(mr_scores))
        mr_max = float(np.max(mr_scores))
        mr_mean = float(np.mean(mr_scores))
        mr_median = float(np.median(mr_scores))

        mag_min = float(np.min(magnitudes))
        mag_max = float(np.max(magnitudes))
        mag_mean = float(np.mean(magnitudes))
        mag_median = float(np.median(magnitudes))

        f1_min = float(np.min(F_front[:, 0]))
        f1_max = float(np.max(F_front[:, 0]))
        f2_min = float(np.min(F_front[:, 1]))
        f2_max = float(np.max(F_front[:, 1]))

        hv_value = compute_hv(F_front, hv_indicator)

        hv_improvement = 0.0
        hv_relative_improvement = 0.0
        if hv_value > hv_best_so_far:
            if np.isfinite(hv_best_so_far) and hv_best_so_far > 0:
                hv_improvement = hv_value - hv_best_so_far
                hv_relative_improvement = hv_improvement / hv_best_so_far
            elif not np.isfinite(hv_best_so_far):
                hv_improvement = hv_value
                hv_relative_improvement = 0.0

            hv_best_so_far = hv_value
            hv_first_gen_at_best = gen

        if first_violation_gen is None and mr_max > 0:
            first_violation_gen = gen

        conv_rows.append({
            "generation": gen,
            "n_eval": n_eval,
            "pareto_size": len(F_front),
            "hv": hv_value,
            "hv_best_so_far": hv_best_so_far,
            "hv_improvement": hv_improvement,
            "hv_relative_improvement": hv_relative_improvement,
            "hv_best_generation_so_far": hv_first_gen_at_best if hv_first_gen_at_best is not None else "",
            "mr_min": mr_min,
            "mr_max": mr_max,
            "mr_mean": mr_mean,
            "mr_median": mr_median,
            "mag_min": mag_min,
            "mag_max": mag_max,
            "mag_mean": mag_mean,
            "mag_median": mag_median,
            "f1_min": f1_min,
            "f1_max": f1_max,
            "f2_min": f2_min,
            "f2_max": f2_max,
            "first_violation_generation_so_far": first_violation_gen if first_violation_gen is not None else ""
        })

        if gen % 25 == 0 or gen == 1:
            print(
                f"Gen {gen:04d} | "
                f"evals={n_eval} | "
                f"Pareto n={len(F_front)} | "
                f"HV={hv_value:.6f} | "
                f"HV*={hv_best_so_far:.6f} | "
                f"MR range=[{mr_min:.4f}, {mr_max:.4f}] | "
                f"Mag range=[{mag_min:.4f}, {mag_max:.4f}]"
            )

    problem = MRGuidedFaceProblem()
    algorithm = NSGA2(pop_size=POP_SIZE)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", N_GEN),
        seed=seed,
        verbose=False,
        callback=callback
    )

    save_csv(image_out / "convergence.csv", conv_rows)

    
    # Final NSGA-II population
    
    if res.pop is None:
        raise RuntimeError("NSGA-II completed without a final population.")

    X_saved = np.asarray(res.pop.get("X"), dtype=float)
    F_saved = np.asarray(res.pop.get("F"), dtype=float)

    if len(X_saved) != POP_SIZE or len(F_saved) != POP_SIZE:
        raise RuntimeError(
            f"Expected exactly {POP_SIZE} final NSGA-II solutions, "
            f"but obtained X={len(X_saved)} and F={len(F_saved)}."
        )

    rank_saved = get_all_ranks(F_saved)
    rank0_indices = np.where(rank_saved == 0)[0]

    
    # Save ALL_SAVED:
    # native final NSGA-II population
    
    np.save(all_saved_dir / "saved_X.npy", X_saved)
    np.save(all_saved_dir / "saved_F.npy", F_saved)

    saved_rows = []

    for i, x in enumerate(X_saved):
        # Re-evaluate only to recover scalar candidate metrics for the CSV.
        # Perturbed images, prediction visualizations, and per-candidate JSON
        # files are intentionally not saved in the compact output format.
        row, _, _ = evaluate_candidate(
            x=x,
            orig_img=orig_img,
            ref_preds=ref_preds,
            gt_boxes=gt_boxes,
            detector=detector
        )

        row = {
            "image_name": img_path.name,
            "candidate_id": i + 1,
            "rank": int(rank_saved[i]),
            **row
        }

        saved_rows.append(row)

    save_csv(all_saved_dir / "saved_candidates.csv", saved_rows)

    with open(all_saved_dir / "saved_meta.json", "w") as f:
        json.dump(json_safe({
            "image": img_path.name,
            "category": CATEGORY,
            "method": "NSGA-II",
            "seed": seed,
            "saved_n": int(len(X_saved)),
            "target_n": int(POP_SIZE),
            "nominal_total_evaluations": POP_SIZE * N_GEN,
            "actual_total_evaluations": (
                None if not conv_rows else conv_rows[-1]["n_eval"]
            ),
            "storage": (
                "compact numeric storage; no per-candidate images/JSON "
                "and no separate ALL_PARETO directory"
            ),
            "selection": (
                "native final NSGA-II population; rank-0 solutions are "
                "identified by rank == 0 in saved_candidates.csv"
            ),
            "rank_counts": {
                str(int(r)): int(np.sum(rank_saved == r))
                for r in np.unique(rank_saved)
            }
        }), f, indent=2)

    if len(saved_rows) != POP_SIZE:
        raise RuntimeError(
            f"Expected to save exactly {POP_SIZE} NSGA-II candidates, "
            f"but saved {len(saved_rows)}."
        )

    best_saved = (
        max(saved_rows, key=lambda r: r["mr_score"])
        if saved_rows else None
    )

    summary = {
        "method": "NSGA-II",
        "category": CATEGORY,
        "image_name": img_path.name,
        "seed": seed,
        "population": POP_SIZE,
        "generations": N_GEN,
        "score_threshold": SCORE_THRESHOLD,
        "pred_match_iou_threshold": PRED_MATCH_IOU_THRESHOLD,
        "num_gt_faces": len(gt_boxes),
        "num_reference_predictions": len(ref_preds),
        "first_violation_generation": first_violation_gen,
        "num_pareto_solutions": int(len(rank0_indices)),
        "num_saved_solutions": len(saved_rows),
        "best_saved_mr_score": None if best_saved is None else best_saved["mr_score"],
        "best_saved_magnitude": None if best_saved is None else best_saved["perturbation_magnitude"],
        "final_hv": None if not conv_rows else conv_rows[-1]["hv"],
        "best_hv": None if not conv_rows else max(r["hv_best_so_far"] for r in conv_rows),
        "best_hv_generation": None if not conv_rows else conv_rows[-1]["hv_best_generation_so_far"],
        "num_saved_detection_violations": sum(r["MR1_detection_preservation_violated"] for r in saved_rows),
        "num_saved_confidence_violations": sum(r["MR2_confidence_stability_violated"] for r in saved_rows),
        "num_saved_localization_violations": sum(r["MR3_localization_stability_violated"] for r in saved_rows),
        "num_saved_mr_score_positive": sum(r["MR_score_positive"] for r in saved_rows),
        "total_lost_predictions_saved": sum(r["lost_predictions"] for r in saved_rows),
        "total_new_predictions_saved": sum(r["new_predictions"] for r in saved_rows),
    }

    with open(image_out / "summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    print(
        f"[done] {img_path.name} | "
        f"best saved MR={summary['best_saved_mr_score']} | "
        f"first violation gen={first_violation_gen}"
    )

    return summary



# Main

def main():
    annotations = load_gt_annotations(GT_FILE, CATEGORY)

    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH,
        config="",
        input_size=(320, 320),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K
    )

    image_files = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    print(f"Category: {CATEGORY}")
    print(f"Images found: {len(image_files)}")
    print(f"Output root: {OUT_ROOT}")

    all_summaries = []

    for img_idx, img_path in enumerate(image_files):
        gt_boxes = annotations.get(img_path.name, [])
        seed = BASE_SEED + img_idx

        image_out = OUT_ROOT / safe_name(img_path)
        summary_path = image_out / "summary.json"

        # Resume support: skip only outputs produced by the current compact
        # NSGA-II configuration. Older summaries do not contain method=NSGA-II
        # and will therefore be reprocessed.
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    existing_summary = json.load(f)

                summary_matches = (
                    existing_summary.get("method") == "NSGA-II"
                    and existing_summary.get("image_name") == img_path.name
                    and existing_summary.get("population") == POP_SIZE
                    and existing_summary.get("generations") == N_GEN
                    and existing_summary.get("score_threshold") == SCORE_THRESHOLD
                    and existing_summary.get("pred_match_iou_threshold")
                    == PRED_MATCH_IOU_THRESHOLD
                )

                if summary_matches:
                    print(f"[resume] Skipping completed image: {img_path.name}")
                    all_summaries.append(existing_summary)
                    continue

                print(
                    f"[warn] Existing summary does not match the current "
                    f"NSGA-II configuration for {img_path.name}. Reprocessing."
                )

            except Exception as e:
                print(
                    f"[warn] Could not read existing summary for "
                    f"{img_path.name}: {e}. Reprocessing."
                )

        try:
            summary = run_one_image(
                img_path=img_path,
                gt_boxes=gt_boxes,
                detector=detector,
                seed=seed
            )

            if summary is not None:
                all_summaries.append(summary)

        except Exception as e:
            print(f"[ERROR] image={img_path.name}: {e}")

    if all_summaries:
        save_csv(OUT_ROOT / "summary_all_images.csv", all_summaries)

    print("\n=== All done ===")
    print(f"Saved summary to: {OUT_ROOT / 'summary_all_images.csv'}")


if __name__ == "__main__":
    main()

