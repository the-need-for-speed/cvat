import json
import os

import requests

SAM2_SERVER = os.environ.get("SAM2_SERVER_URL", "http://172.18.0.1:8091")


def init_context(context):
    context.logger.info(f"SAM2 proxy init — server: {SAM2_SERVER}")
    try:
        r = requests.get(f"{SAM2_SERVER}/health", timeout=5)
        context.logger.info(f"SAM2 server health: {r.json()}")
    except Exception as e:
        context.logger.warn(f"SAM2 server not reachable at init: {e}")


def handler(context, event):
    context.logger.info("SAM2 handler called")
    data = event.body

    payload = {
        "image":      data["image"],
        "pos_points": data.get("pos_points") or [],
        "neg_points": data.get("neg_points") or [],
        "obj_bbox":   data.get("obj_bbox"),
        "threshold":  float(data.get("threshold", 0.0)),
    }

    resp = requests.post(
        f"{SAM2_SERVER}/segment_sam2_cvat",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )
