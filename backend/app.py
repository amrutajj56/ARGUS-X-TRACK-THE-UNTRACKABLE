from __future__ import annotations
import csv, os, uuid, threading, traceback, time
from pathlib import Path
from typing import Any
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
DATA.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app = FastAPI(title="ARGUS-X", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173"
], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.RLock()
CAMERA = {"capture": None, "engine": None, "identity": None, "active": False}
CAMERA_LOCK = threading.RLock()


@app.get("/api/health")
def health():
    return {"status": "online", "platform": "ARGUS-X"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}:
        raise HTTPException(400, "Please upload an MP4, AVI, MOV, MKV, M4V or WEBM video.")
    job_id = uuid.uuid4().hex[:12]
    dest = DATA / f"{job_id}{suffix}"
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > 500 * 1024 * 1024:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "Video exceeds the 500 MB limit.")
            out.write(chunk)
    with LOCK:
        JOBS[job_id] = {
            "job_id": job_id, "filename": file.filename, "source": str(dest),
            "status": "uploaded", "progress": 0, "message": "Video ready",
            "stats": {}, "error": None, "events": [], "identities": []
        }
    return {"job_id": job_id, "filename": file.filename, "size_bytes": size}


@app.post("/api/process/{job_id}")
def process_video(job_id: str, background_tasks: BackgroundTasks):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Upload not found. Please upload the video again.")
        if job["status"] == "processing":
            raise HTTPException(409, "This video is already processing.")
        job.update(status="processing", progress=1, message="Starting analysis", error=None)
    background_tasks.add_task(_run_video_job, job_id)
    return {"job_id": job_id, "status": "processing"}


def _run_video_job(job_id: str):
    cap = writer = None
    try:
        from src.tracker import TrackingEngine
        from src.identity import GlobalIdentityManager
        job = JOBS[job_id]
        cap = cv2.VideoCapture(job["source"])
        if not cap.isOpened():
            raise RuntimeError("Could not read the uploaded video. Try another MP4/AVI file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 1: fps = 25.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        out_path, csv_path = OUTPUTS / f"{job_id}_tracked.mp4", OUTPUTS / f"{job_id}_tracks.csv"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create the annotated output video.")
        engine = TrackingEngine(
            model_path=os.getenv("ARGUS_MODEL", "yolov8n.pt"),
            confidence=float(os.getenv("ARGUS_CONFIDENCE", "0.35")), device="cpu"
        )
        identity = GlobalIdentityManager(
            similarity_threshold=float(os.getenv("ARGUS_REID_THRESHOLD", "0.82")),
            max_lost_frames=int(os.getenv("ARGUS_ID_MEMORY_FRAMES", "900"))
        )
        fields = ["frame", "camera_id", "local_track_id", "global_id", "class_id",
                  "class_name", "confidence", "identity_match_status", "match_similarity",
                  "lifecycle_status", "x1", "y1", "x2", "y2"]
        count = observations = 0
        local_ids, global_ids, classes = set(), set(), {}
        events = []
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fields)
            csv_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok: break
                annotated, raw_tracks = engine.process_frame(frame)
                tracks = identity.assign_frame(frame, raw_tracks)
                count += 1
                for t in tracks:
                    x1, y1, x2, y2 = t["bbox"]
                    gid, tid = t["global_id"], t.get("track_id")
                    color = (50, 220, 170)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f'{t["class_name"]} | GID {gid}',
                                (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, color, 2, cv2.LINE_AA)
                    row = {
                        "frame": count, "camera_id": "camera_1", "local_track_id": tid,
                        "global_id": gid, "class_id": t["class_id"], "class_name": t["class_name"],
                        "confidence": round(float(t["confidence"]), 4),
                        "identity_match_status": t["identity_status"],
                        "match_similarity": t["match_similarity"],
                        "lifecycle_status": t["lifecycle_status"],
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    }
                    csv_writer.writerow(row)
                    observations += 1
                    if tid is not None: local_ids.add(int(tid))
                    global_ids.add(int(gid))
                    classes[t["class_name"]] = classes.get(t["class_name"], 0) + 1
                    if len(events) < 1000:
                        events.append(row)
                writer.write(annotated)
                if total and count % 3 == 0:
                    with LOCK:
                        JOBS[job_id].update(
                            progress=min(99, int(count / total * 100)),
                            message=f"Analyzing frame {count:,} of {total:,}"
                        )
        if count == 0:
            raise RuntimeError("No readable frames were found.")
        identities = identity.summary()
        stats = {
            "frames_processed": count, "track_observations": observations,
            "unique_local_ids": len(local_ids), "unique_global_ids": len(global_ids),
            "classes": classes, "fps": round(fps, 2),
            "resolution": f"{width} × {height}",
            "identity_mode": "Appearance-based identity memory (best-effort; ambiguous matches are not forced)"
        }
        with LOCK:
            JOBS[job_id].update(
                status="completed", progress=100, message="Analysis completed",
                stats=stats, events=events, identities=identities,
                video_url=f"/api/video/{job_id}", csv_url=f"/api/csv/{job_id}"
            )
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            JOBS[job_id].update(status="failed", progress=0, message="Analysis failed", error=str(exc))
    finally:
        if cap is not None: cap.release()
        if writer is not None: writer.release()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "Job not found")
        return {k: v for k, v in job.items() if k != "source"}


@app.get("/api/events/{job_id}")
def get_events(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "Job not found")
        return {"events": job.get("events", [])}


@app.get("/api/identities/{job_id}")
def get_identities(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "Job not found")
        return {"identities": job.get("identities", [])}


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    path = OUTPUTS / f"{job_id}_tracked.mp4"
    if not path.exists(): raise HTTPException(404, "Processed video is not ready")
    return FileResponse(path, media_type="video/mp4", filename="argusx_annotated_tracking.mp4")


@app.get("/api/csv/{job_id}")
def get_csv(job_id: str):
    path = OUTPUTS / f"{job_id}_tracks.csv"
    if not path.exists(): raise HTTPException(404, "Tracking CSV is not ready")
    return FileResponse(path, media_type="text/csv", filename="argusx_tracking_report.csv")


@app.post("/api/webcam/start")
def start_webcam():
    with CAMERA_LOCK:
        if CAMERA["active"]:
            return {"status": "already_running"}
        from src.tracker import TrackingEngine
        from src.identity import GlobalIdentityManager
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            raise HTTPException(503, "Could not open the webcam. Check Windows camera permissions.")
        try:
            engine = TrackingEngine(
                model_path=os.getenv("ARGUS_MODEL", "yolov8n.pt"),
                confidence=float(os.getenv("ARGUS_CONFIDENCE", "0.35")), device="cpu"
            )
            identity = GlobalIdentityManager()
        except Exception:
            cap.release()
            raise
        CAMERA.update(capture=cap, engine=engine, identity=identity, active=True)
    return {"status": "started"}


@app.post("/api/webcam/stop")
def stop_webcam():
    with CAMERA_LOCK:
        CAMERA["active"] = False
        cap = CAMERA.get("capture")
        CAMERA.update(capture=None, engine=None, identity=None)
        if cap is not None: cap.release()
    return {"status": "stopped"}


@app.get("/api/webcam/stream")
def webcam_stream():
    def generate():
        while True:
            with CAMERA_LOCK:
                if not CAMERA["active"]: break
                cap, engine, identity = CAMERA["capture"], CAMERA["engine"], CAMERA["identity"]
                if cap is None: break
                ok, frame = cap.read()
                if not ok: break
                try:
                    annotated, raw_tracks = engine.process_frame(frame)
                    tracks = identity.assign_frame(frame, raw_tracks)
                    for track in tracks:
                        x1, y1, x2, y2 = track["bbox"]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 170), 2)
                        cv2.putText(annotated, f'GID {track["global_id"]}',
                                    (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (50, 220, 170), 2, cv2.LINE_AA)
                    ok, encoded = cv2.imencode(".jpg", annotated)
                    if not ok: continue
                    payload = encoded.tobytes()
                except Exception:
                    traceback.print_exc()
                    break
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(0.025)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def root():
    return {"name": "ARGUS-X", "health": "/api/health", "docs": "/docs"}
