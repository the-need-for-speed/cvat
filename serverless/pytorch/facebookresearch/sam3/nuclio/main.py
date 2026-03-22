# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

import base64
import io
import json
import os

import requests
from PIL import Image

SAM3_SERVER = os.environ.get("SAM3_SERVER_URL", "http://172.18.0.1:8091")
MAX_SIDE = int(os.environ.get("SAM3_MAX_SIDE", "1280"))


def _resize_image_b64(image_b64: str) -> tuple[str, float, float]:
    """Resize image so its longest side is at most MAX_SIDE.

    Returns (resized_b64, scale_x, scale_y) where scale_* are the factors to
    multiply SAM3's output coordinates by to get back to original pixel coords.
    """
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    longest = max(orig_w, orig_h)
    if longest <= MAX_SIDE:
        return image_b64, 1.0, 1.0

    scale = MAX_SIDE / longest
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    resized_b64 = base64.b64encode(buf.getvalue()).decode()

    scale_x = orig_w / new_w
    scale_y = orig_h / new_h
    return resized_b64, scale_x, scale_y


def _scale_points(points: list[float], scale_x: float, scale_y: float) -> list[float]:
    return [
        v * scale_x if i % 2 == 0 else v * scale_y
        for i, v in enumerate(points)
    ]


def init_context(context):
    context.logger.info(f"SAM3 proxy init — server: {SAM3_SERVER}, max_side: {MAX_SIDE}")
    try:
        r = requests.get(f"{SAM3_SERVER}/health", timeout=5)
        context.logger.info(f"SAM3 server health: {r.json()}")
    except Exception as e:
        context.logger.warn(f"SAM3 server not reachable at init: {e}")


def handler(context, event):
    context.logger.info("SAM3 handler called")
    data = event.body
    image_b64 = data["image"]
    threshold = float(data.get("threshold", 0.5))

    # model_labels contains the model spec label names from the user's mapping.
    # Each label name is used as the SAM3 text prompt.
    model_labels = data.get("model_labels", [])

    if not model_labels:
        context.logger.warn("No model_labels received — nothing to detect")
        return context.Response(
            body=json.dumps([]),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    output_type = data.get("output_type", "polygon")

    resized_b64, scale_x, scale_y = _resize_image_b64(image_b64)
    context.logger.info(
        f"Image resize: scale_x={scale_x:.3f}, scale_y={scale_y:.3f} "
        f"(max_side={MAX_SIDE}), output_type={output_type}"
    )

    all_results = []
    for prompt in model_labels:
        context.logger.info(f"Running SAM3 with prompt: '{prompt}'")
        try:
            resp = requests.post(
                f"{SAM3_SERVER}/segment_cvat",
                json={"image": resized_b64, "prompt": prompt, "confidence": threshold},
                timeout=120,
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            context.logger.error(f"SAM3 request failed for prompt '{prompt}': {e}")
            continue

        for r in results:
            r["label"] = prompt
            if scale_x != 1.0 or scale_y != 1.0:
                r["points"] = _scale_points(r["points"], scale_x, scale_y)
            if output_type == "rectangle":
                pts = r["points"]
                xs = pts[0::2]
                ys = pts[1::2]
                r["points"] = [min(xs), min(ys), max(xs), max(ys)]
                r["type"] = "rectangle"
        all_results.extend(results)
        context.logger.info(f"SAM3 found {len(results)} instances for '{prompt}'")

    return context.Response(
        body=json.dumps(all_results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
