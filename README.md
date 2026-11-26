# ARGUS-X — Integrated Vision Intelligence Platform

This build includes a React command center, FastAPI service, YOLO/ByteTrack short-term tracking, a CPU-friendly appearance-based global identity memory, annotated video output, CSV reporting, event and identity views, and webcam streaming.

## Important identity behaviour
- Each analysis starts a new identity session.
- A global ID is assigned independently of the temporary local tracker ID.
- When a track fragment disappears and a visually similar same-class object reappears within the configured memory window, the appearance descriptor can reconnect it to its previous global ID.
- The included appearance descriptor uses HSV colour histograms split into horizontal body regions. It is a practical CPU baseline, **not a trained DINO/ReID neural network**. Similar clothing, lighting changes, camera changes, low-resolution crops, and occlusion can cause uncertain matches. The system deliberately avoids forcing ambiguous matches.
- Default memory window is 900 frames. Tune `ARGUS_REID_THRESHOLD` (default 0.82) and `ARGUS_ID_MEMORY_FRAMES` if validating on your own footage. Always evaluate with the actual demo video; no appearance-only method can guarantee identity correctness in every scene.

## Windows setup

### 1. Backend terminal
From the project root:

```powershell
.\start_backend.bat
```

First run installs Python dependencies and downloads the YOLO nano weights if not already cached.

### 2. Frontend terminal
Open a second PowerShell window:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open http://localhost:5173

If PowerShell blocks `npm`, use `npm.cmd` as shown. Keep both terminals open.

## Output
After processing a video, use **Export CSV Report** to download frame-level events. The annotated MP4 is available from the preview. CSV columns include frame, camera ID, local track ID, global ID, class, confidence, identity status, similarity, lifecycle status, and bounding box.

## Webcam
Use Start Live Webcam in the command center. Stop it before closing the application. Windows camera access must be enabled.

## Troubleshooting
- API offline: keep the backend terminal open and visit http://127.0.0.1:8000/api/health.
- Port 8000 in use: run `netstat -ano | findstr :8000` and identify the existing process before stopping it.
- Webcam unavailable: close other apps using the camera and check Windows Settings > Privacy & security > Camera.
- First YOLO run can take time while weights are downloaded.
