"""Lightweight appearance-based global identity memory for ARGUS-X.

This is a CPU-friendly baseline ReID descriptor built from HSV colour histograms.
It is not a learned person-recognition model. Similar-looking people can still be
ambiguous; uncertain matches are assigned a new identity rather than forced.
"""
from __future__ import annotations
from collections import deque
import cv2
import numpy as np


class GlobalIdentityManager:
    def __init__(self, similarity_threshold: float = 0.82,
                 ambiguity_margin: float = 0.035,
                 max_lost_frames: int = 900,
                 max_features: int = 8,
                 local_mapping_max_age: int = 120):
        self.similarity_threshold = similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        self.max_lost_frames = max_lost_frames
        self.max_features = max_features
        self.local_mapping_max_age = local_mapping_max_age
        self.next_global_id = 1
        self.identities = {}
        self.local_map = {}
        self.frame_number = 0

    @staticmethod
    def _feature(frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 12:
            return None
        crop = frame[y1:y2, x1:x2]
        # Reduce background contribution and standardize descriptor geometry.
        ch, cw = crop.shape[:2]
        crop = crop[int(ch * 0.08):max(int(ch * 0.92), 1),
                    int(cw * 0.12):max(int(cw * 0.88), 1)]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        parts = np.array_split(hsv, 4, axis=0)
        pieces = []
        for part in parts:
            hist = cv2.calcHist([part], [0, 1], None, [24, 16], [0, 180, 0, 256])
            hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()
            pieces.append(hist)
        vector = np.concatenate(pieces).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None

    @staticmethod
    def _similarity(a, b):
        return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))

    def assign_frame(self, frame, tracks):
        self.frame_number += 1
        frame_no = self.frame_number
        assigned_this_frame = set()
        output = []

        for track in tracks:
            track = dict(track)
            local_id = track.get("track_id")
            class_name = str(track.get("class_name", "object"))
            local_key = (class_name, int(local_id)) if local_id is not None else None
            feature = self._feature(frame, track["bbox"])
            mapped_gid = self.local_map.get(local_key) if local_key else None
            identity = self.identities.get(mapped_gid) if mapped_gid else None

            # A ByteTrack local ID can persist through short gaps. Do not let a
            # stale mapping collide with a global ID already assigned this frame.
            if identity and frame_no - identity["last_seen"] <= self.local_mapping_max_age \
                    and mapped_gid not in assigned_this_frame:
                gap = frame_no - identity["last_seen"]
                status = "known_local_track" if gap <= 1 else "recovered_local_track"
                similarity = None
                gid = mapped_gid
            else:
                if local_key:
                    self.local_map.pop(local_key, None)
                gid, similarity = self._match(feature, class_name, assigned_this_frame, frame_no)
                if gid is not None:
                    status = "appearance_matched"
                else:
                    gid = self.next_global_id
                    self.next_global_id += 1
                    status = "new_identity"
                    self.identities[gid] = {
                        "class_name": class_name,
                        "features": deque(maxlen=self.max_features),
                        "first_seen": frame_no,
                        "last_seen": frame_no,
                        "observations": 0,
                    }

            identity = self.identities[gid]
            if feature is not None and (status in ("new_identity", "appearance_matched") or
                                        identity["observations"] % 15 == 0):
                identity["features"].append(feature)
            identity["last_seen"] = frame_no
            identity["observations"] += 1
            assigned_this_frame.add(gid)
            if local_key:
                self.local_map[local_key] = gid

            track["global_id"] = gid
            track["identity_status"] = status
            track["match_similarity"] = round(similarity, 4) if similarity is not None else ""
            track["lifecycle_status"] = "recovered" if "recovered" in status or status == "appearance_matched" else "active"
            output.append(track)

        # Expire old local mappings, but retain global identity galleries for the
        # configured reappearance window.
        for key, gid in list(self.local_map.items()):
            identity = self.identities.get(gid)
            if identity is None or frame_no - identity["last_seen"] > self.local_mapping_max_age:
                self.local_map.pop(key, None)
        for gid, identity in list(self.identities.items()):
            if frame_no - identity["last_seen"] > self.max_lost_frames:
                # Keep a small dormant memory longer than local tracker mappings.
                identity["dormant"] = True
        return output

    def _match(self, feature, class_name, assigned, frame_no):
        if feature is None:
            return None, None
        ranked = []
        for gid, identity in self.identities.items():
            if gid in assigned or identity["class_name"] != class_name:
                continue
            gap = frame_no - identity["last_seen"]
            if gap < 1 or gap > self.max_lost_frames or not identity["features"]:
                continue
            score = max(self._similarity(feature, old) for old in identity["features"])
            ranked.append((score, gid))
        if not ranked:
            return None, None
        ranked.sort(reverse=True)
        best_score, best_gid = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0
        if best_score >= self.similarity_threshold and best_score - second_score >= self.ambiguity_margin:
            return best_gid, best_score
        return None, best_score

    def summary(self):
        return [
            {"global_id": gid, "class_name": item["class_name"],
             "first_seen_frame": item["first_seen"], "last_seen_frame": item["last_seen"],
             "observations": item["observations"]}
            for gid, item in sorted(self.identities.items())
        ]
