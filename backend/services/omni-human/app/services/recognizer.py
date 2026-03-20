
import logging
import threading
import numpy as np

logger = logging.getLogger("omni-human.recognizer")


class FaceRecognizer:
    """
    InsightFace-based face recognizer.
    Uses buffalo_l model for state-of-the-art face detection, alignment, and embedding.
    
    Improvements:
    - Graceful GPU→CPU fallback
    - Configurable det_size for different resolution inputs
    - Min face size filter to avoid false positives
    - Embedding normalization check
    - Thread-safe: InsightFace model is NOT thread-safe, so inference is serialized
    """

    MIN_FACE_SIZE = 20  # Skip faces smaller than 20x20 px
    
    def __init__(self, root_dir: str, model_name: str = 'buffalo_l', use_gpu: bool = True):
        self.root_dir = root_dir
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.app = None
        self._lock = threading.Lock()  # InsightFace is NOT thread-safe
        
        # Lazy import to avoid import errors when insightface not installed
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.error("❌ insightface not installed. FaceRecognizer disabled.")
            return
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        logger.info("Loading InsightFace (%s) from %s...", model_name, root_dir)
        try:
            ctx_id = 0 if use_gpu else -1
            self.app = FaceAnalysis(name=model_name, root=root_dir, providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("✅ Face Recognizer loaded (GPU=%s)", use_gpu)
        except Exception as e:
            logger.warning("⚠️ GPU load failed: %s — retrying with CPU...", e)
            if use_gpu:
                try:
                    self.app = FaceAnalysis(name=model_name, root=root_dir, providers=['CPUExecutionProvider'])
                    self.app.prepare(ctx_id=-1, det_size=(640, 640))
                    logger.info("✅ Face Recognizer loaded (CPU fallback)")
                except Exception as ex:
                    logger.error("❌ Fatal: Face Recognizer failed to load: %s", ex)
                    self.app = None
            else:
                self.app = None

    def extract_embedding(self, image: np.ndarray, min_face_size: int = None, min_det_score: float = None):
        """
        Extract embeddings for all faces in the image.
        
        Uses InsightFace's full pipeline: detect → align → extract embedding.
        
        Args:
            image: BGR numpy array (H, W, 3) — OpenCV / InsightFace convention.
                   If the caller has RGB data, convert with ``cv2.cvtColor(img, cv2.COLOR_RGB2BGR)``
                   before calling this method.
            min_face_size: Minimum face width/height in pixels (default: self.MIN_FACE_SIZE)
            min_det_score: Minimum detection score to accept (default: from config)
            
        Returns:
            List of dicts with: bbox, kps, embedding, score
        """
        if self.app is None:
            return []

        min_size = min_face_size or self.MIN_FACE_SIZE
        if min_det_score is None:
            from app.config import get_settings
            min_det_score = get_settings().insightface_det_score
        
        try:
            with self._lock:
                faces = self.app.get(image)
        except Exception as e:
            logger.warning("InsightFace inference error: %s", e)
            return []

        results = []
        for face in faces:
            # Skip tiny faces (likely false positives)
            fw = face.bbox[2] - face.bbox[0]
            fh = face.bbox[3] - face.bbox[1]
            if fw < min_size or fh < min_size:
                continue

            # Skip low-confidence detections
            if face.det_score < min_det_score:
                continue

            # Validate embedding
            if face.embedding is None or len(face.embedding) == 0:
                continue

            embedding = face.embedding.tolist()
            
            # Sanity check: embedding should be normalized (L2 norm ≈ 1.0)
            norm = np.linalg.norm(face.embedding)
            if norm < 0.1:  # Degenerate embedding
                logger.warning("Skipping degenerate embedding (norm=%.3f)", norm)
                continue

            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "kps": face.kps.astype(int).tolist() if face.kps is not None else None,
                "embedding": embedding,
                "score": float(face.det_score),
            })

        return results

    def compare_faces(self, embedding1, embedding2):
        """
        Compute cosine similarity between two face embeddings.
        InsightFace embeddings are normalized, so dot product = cosine similarity.
        
        Returns:
            float: similarity score (-1.0 to 1.0). Higher = more similar.
            Typical threshold: 0.4-0.6 for same person.
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        e1 = np.array(embedding1, dtype=np.float32)
        e2 = np.array(embedding2, dtype=np.float32)
        
        # Normalize for safety (should already be normalized)
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        
        return float(np.dot(e1 / n1, e2 / n2))
