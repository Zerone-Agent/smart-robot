from typing import List, Dict, Any, Optional


def calculate_priority_score(
    detection: Dict[str, Any], weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate priority score for a tomato detection."""
    if weights is None:
        weights = {
            "confidence": 0.3,
            "maturity": 0.3,
            "accessibility": 0.2,
            "distance": 0.2,
        }
    confidence = detection.get("confidence", 1.0)
    maturity = detection.get("maturity_score", 0.0)
    accessibility = detection.get("accessibility", 1.0)
    distance = detection.get("distance", 0.0)
    # Normalize distance: closer is better, so invert it
    max_distance = detection.get("max_distance", 1000.0)
    if max_distance <= 0:
        distance_score = 1.0
    else:
        distance_score = max(0, 1.0 - min(distance / max_distance, 1.0))
    score = (
        weights["confidence"] * confidence
        + weights["maturity"] * maturity
        + weights["accessibility"] * accessibility
        + weights["distance"] * distance_score
    )
    return score


def sort_by_priority(
    detections: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """Sort detections by priority score in descending order."""
    scored = []
    for det in detections:
        score = calculate_priority_score(det, weights)
        det_copy = det.copy()
        det_copy["priority_score"] = score
        scored.append(det_copy)
    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored
