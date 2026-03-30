#!/usr/bin/env python3
"""
Extract the N most visually distinct frames from a set of videos.

Uses CLIP embeddings + farthest-point sampling to maximise diversity
among selected frames, ensuring annotation effort is spent on unique
and informative images.

Usage:
    # From a directory of videos
    python extract_valuable_frames.py /path/to/videos/ -o /path/to/frames/ -n 500

    # From specific video files
    python extract_valuable_frames.py vid1.mp4 vid2.mp4 -o /path/to/frames/ -n 200

    # With custom settings
    python extract_valuable_frames.py /path/to/videos/ -o /path/to/frames/ \
        -n 500 --sample-interval 0.5 --blur-threshold 80 --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".m4v", ".webm"}


def log(msg="", flush=True):
    print(msg, flush=flush)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def find_videos(paths):
    """Given a list of paths (files or directories), return all video files."""
    videos = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS and not f.name.startswith("._"):
                    videos.append(f)
    return videos


def is_blurry(frame_gray, threshold):
    return cv2.Laplacian(frame_gray, cv2.CV_64F).var() < threshold


def is_too_dark(frame_gray, threshold=20):
    return frame_gray.mean() < threshold


def extract_candidates(videos, sample_interval, blur_threshold):
    """Extract candidate frame metadata from all videos."""
    candidates = []
    t_stage = time.monotonic()

    for vid_i, video_path in enumerate(videos):
        t_vid = time.monotonic()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"  WARN: cannot open {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        frame_step = max(1, int(fps * sample_interval))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        frame_idx = 0
        next_sample = 0
        extracted = 0
        rejected = 0

        while True:
            if frame_idx < next_sample:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if blur_threshold > 0 and is_blurry(gray, blur_threshold):
                rejected += 1
                frame_idx += 1
                next_sample = frame_idx + frame_step - 1
                continue

            if is_too_dark(gray):
                rejected += 1
                frame_idx += 1
                next_sample = frame_idx + frame_step - 1
                continue

            candidates.append({
                "video": str(video_path),
                "video_stem": video_path.stem,
                "video_idx": vid_i,
                "frame_idx": frame_idx,
                "timestamp": round(frame_idx / fps, 3),
                "fps": fps,
            })
            extracted += 1
            frame_idx += 1
            next_sample = frame_idx + frame_step - 1

        cap.release()
        elapsed = time.monotonic() - t_vid
        log(f"  [{vid_i + 1}/{len(videos)}] {video_path.name} "
            f"({duration:.0f}s video): {extracted} candidates, "
            f"{rejected} rejected [{elapsed:.1f}s]")

    elapsed_total = time.monotonic() - t_stage
    log(f"  Stage complete: {len(candidates)} total candidates from "
        f"{len(videos)} videos [{elapsed_total:.1f}s]")
    return candidates


# ---------------------------------------------------------------------------
# CLIP embedding
# ---------------------------------------------------------------------------

def compute_clip_embeddings(candidates, device, batch_size=64):
    """Compute CLIP ViT-B/32 embeddings, reading frames from disk on the fly."""
    import clip
    from PIL import Image

    log(f"  Loading CLIP ViT-B/32 on {device}...")
    t_load = time.monotonic()
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    log(f"  CLIP loaded [{time.monotonic() - t_load:.1f}s]")

    n = len(candidates)
    embed_dim = 512
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)

    video_groups = defaultdict(list)
    for i, c in enumerate(candidates):
        video_groups[c["video"]].append((i, c["frame_idx"]))

    processed = 0
    n_videos = len(video_groups)
    t_stage = time.monotonic()

    for vid_num, (video_path, frame_list) in enumerate(video_groups.items()):
        frame_list.sort(key=lambda x: x[1])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        batch_indices = []
        batch_tensors = []

        for cand_idx, frame_idx in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            batch_tensors.append(preprocess(pil_img))
            batch_indices.append(cand_idx)

            if len(batch_tensors) >= batch_size:
                with torch.no_grad():
                    batch = torch.stack(batch_tensors).to(device)
                    features = model.encode_image(batch)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_np = features.cpu().numpy()
                for j, ci in enumerate(batch_indices):
                    embeddings[ci] = features_np[j]
                processed += len(batch_indices)
                batch_indices = []
                batch_tensors = []

        # Flush remaining
        if batch_tensors:
            with torch.no_grad():
                batch = torch.stack(batch_tensors).to(device)
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                features_np = features.cpu().numpy()
            for j, ci in enumerate(batch_indices):
                embeddings[ci] = features_np[j]
            processed += len(batch_indices)

        cap.release()
        elapsed = time.monotonic() - t_stage
        fps_rate = processed / elapsed if elapsed > 0 else 0
        log(f"  [{vid_num + 1}/{n_videos}] Embedded {processed}/{n} frames "
            f"[{fps_rate:.0f} frames/s, {elapsed:.1f}s elapsed]")

    del model
    torch.cuda.empty_cache()

    elapsed_total = time.monotonic() - t_stage
    log(f"  Stage complete: {processed} embeddings [{elapsed_total:.1f}s]")
    return embeddings


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------

def farthest_point_sampling(embeddings, n):
    """Select n indices maximising minimum pairwise distance (greedy)."""
    m = len(embeddings)
    if n >= m:
        return list(range(m))

    t_stage = time.monotonic()
    centroid = embeddings.mean(axis=0)
    dists_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    seed = int(np.argmin(dists_to_centroid))

    selected = [seed]
    min_dists = np.linalg.norm(embeddings - embeddings[seed], axis=1)

    for step in range(1, n):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

        new_dists = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

        for s in selected:
            min_dists[s] = -1

        if (step + 1) % 50 == 0:
            elapsed = time.monotonic() - t_stage
            log(f"  Selected {step + 1}/{n} frames [{elapsed:.1f}s]")

    elapsed_total = time.monotonic() - t_stage
    log(f"  Stage complete: {n} frames selected [{elapsed_total:.1f}s]")
    return selected


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_selected_frames(candidates, selected_indices, output_dir, quality):
    """Re-read selected frames from video and save as JPEGs."""
    t_stage = time.monotonic()
    save_groups = defaultdict(list)
    for rank, idx in enumerate(selected_indices):
        cand = candidates[idx]
        save_groups[cand["video"]].append((rank, idx, cand))

    manifest = []
    saved = 0
    n_videos = len(save_groups)

    for vid_num, (video_path, items) in enumerate(save_groups.items()):
        items.sort(key=lambda x: x[2]["frame_idx"])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        for rank, idx, cand in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand["frame_idx"])
            ret, frame = cap.read()
            if not ret:
                continue
            filename = f"{cand['video_stem']}__frame_{cand['frame_idx']:06d}.jpg"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            manifest.append({
                "filename": filename,
                "video": cand["video"],
                "frame_idx": cand["frame_idx"],
                "timestamp": cand["timestamp"],
                "selection_rank": rank,
            })
            saved += 1
        cap.release()

        if (vid_num + 1) % 10 == 0 or vid_num + 1 == n_videos:
            elapsed = time.monotonic() - t_stage
            log(f"  Saved from {vid_num + 1}/{n_videos} videos "
                f"({saved} frames) [{elapsed:.1f}s]")

    elapsed_total = time.monotonic() - t_stage
    log(f"  Stage complete: {saved} frames saved [{elapsed_total:.1f}s]")
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract the N most visually distinct frames from videos")
    parser.add_argument("inputs", nargs="+",
                        help="Video files or directories containing videos")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for extracted frames")
    parser.add_argument("-n", type=int, default=500,
                        help="Number of frames to extract (default: 500)")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="Seconds between candidate frames (default: 1.0)")
    parser.add_argument("--blur-threshold", type=float, default=50.0,
                        help="Laplacian variance threshold for blur rejection "
                             "(0 to disable, default: 50)")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device for CLIP (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="CLIP batch size (default: 64)")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG quality for saved frames (default: 92)")
    args = parser.parse_args()

    t_total = time.monotonic()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find videos
    log("=" * 60)
    log("STEP 1: Finding videos")
    log("=" * 60)
    videos = find_videos(args.inputs)
    if not videos:
        log("No videos found.")
        return
    log(f"Found {len(videos)} video(s)\n")

    # Step 2: Extract candidate frames
    log("=" * 60)
    log("STEP 2: Extracting candidate frames")
    log(f"  interval={args.sample_interval}s, blur_threshold={args.blur_threshold}")
    log("=" * 60)
    candidates = extract_candidates(videos, args.sample_interval, args.blur_threshold)
    log(f"\n{len(candidates)} candidate frames total\n")

    if not candidates:
        log("No candidate frames found.")
        return

    if args.n >= len(candidates):
        log(f"Requested {args.n} frames but only {len(candidates)} candidates — using all.\n")
        selected_indices = list(range(len(candidates)))
    else:
        # Step 3: Compute CLIP embeddings
        log("=" * 60)
        log("STEP 3: Computing CLIP embeddings")
        log("=" * 60)
        embeddings = compute_clip_embeddings(candidates, args.device, args.batch_size)
        log(f"Embedding matrix: {embeddings.shape}\n")

        # Step 4: Farthest-point sampling
        log("=" * 60)
        log(f"STEP 4: Selecting {args.n} most diverse frames")
        log("=" * 60)
        selected_indices = farthest_point_sampling(embeddings, args.n)
        log()

    # Step 5: Save selected frames
    log("=" * 60)
    log(f"STEP 5: Saving {len(selected_indices)} frames")
    log("=" * 60)
    manifest = save_selected_frames(candidates, selected_indices, output_dir, args.quality)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "n_selected": len(manifest),
            "n_candidates": len(candidates),
            "n_videos": len(videos),
            "settings": {
                "sample_interval": args.sample_interval,
                "blur_threshold": args.blur_threshold,
                "n_requested": args.n,
            },
            "frames": manifest,
        }, f, indent=2)

    elapsed_total = time.monotonic() - t_total
    log()
    log("=" * 60)
    log(f"DONE in {elapsed_total:.1f}s")
    log(f"  {len(manifest)} frames saved to {output_dir}")
    log(f"  Manifest: {manifest_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
