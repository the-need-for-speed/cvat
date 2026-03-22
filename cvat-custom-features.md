# CVAT Custom Features — Implementation Notes

## SAM3 Detector

SAM3 ("Segment Anything with Concepts") is a custom open-vocabulary instance segmentation detector integrated into CVAT as a Nuclio serverless function.

### Architecture

```
Browser → CVAT API → Nuclio (pth-facebookresearch-sam3) → SAM3 HTTP server (port 8091)
```

- **Nuclio function**: `serverless/pytorch/facebookresearch/sam3/nuclio/`
  - `function-gpu.yaml` — defines function metadata, type: `detector`, spec: one model label `text_prompt` of type `polygon`
  - `main.py` — proxy handler: resizes image, sends to SAM3 server, scales results back to original resolution

- **SAM3 HTTP server**: runs separately on `http://172.18.0.1:8091` (the Docker host bridge IP)
  - Endpoint: `POST /segment_cvat` — accepts `{image, prompt, confidence}`, returns `[{points, confidence}]`
  - Endpoint: `GET /health`

- **UI panel** (`cvat-ui/src/components/annotation-page/standard-workspace/objects-side-bar/sam3-panel.tsx`):
  - Fetches the current frame directly from CVAT's API (`/api/jobs/{id}/data?type=frame&quality=original&number={n}`)
  - Posts directly to `/sam3/segment_cvat` (Traefik routes this to the SAM3 server)
  - Allows selecting label, output type (polygon or bounding box), confidence threshold, and free-text prompt
  - On success, dispatches `createAnnotationsAsync` with new ObjectState instances

### How the detector mapping works

The CVAT detector dialog maps **CVAT labels → model labels**. For SAM3, the single model label is `text_prompt`. The label name the user maps to it (e.g. "person") becomes the text prompt sent to SAM3. This means CVAT labels function as text prompts automatically — no separate prompt entry needed when using the standard detector dialog.

### Image resizing

The Nuclio proxy resizes images so the longest side is ≤ 1280px (configurable via `SAM3_MAX_SIDE` env var) before sending to SAM3, then scales the returned polygon coordinates back to original resolution.

---

## Video Review Workspace

A custom CVAT workspace for reviewing video clips and recording a frame count (number of visible objects).

### Files

- `cvat-ui/src/components/annotation-page/video-review-workspace/video-review-workspace.tsx` — workspace component
- `cvat-ui/src/reducers/index.ts` — `Workspace.VIDEO_REVIEW = 'Video review'` enum value

### Behaviour

- Renders a simple UI: current frame count display, a number input, save button — no annotation canvas
- On mount, calls `ensureLabel()` which finds or creates a label named `"Video Review"` (type: `tag`) with a numeric attribute `object_count` on the project or task
- Captures the **spacebar** key globally (in capture phase) to prevent video playback toggle while in this workspace
- Saves annotations as tag objects with the `object_count` attribute set to the user-entered number