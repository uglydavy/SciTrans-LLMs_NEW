"""YOLO-based layout detection for documents."""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import YOLO dependencies
HAS_ULTRALYTICS = False
HAS_TORCH = False
HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    pass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_yolo_model(model_path: Optional[str] = None, device: Optional[str] = None) -> Optional[Any]:
    """
    Load YOLO model for document layout detection.
    
    Args:
        model_path: Path to YOLO model file (.pt). If None, uses default or downloads.
        device: Device to use ('cuda', 'cpu', 'mps'). Auto-detects if None.
    
    Returns:
        YOLO model instance or None if unavailable
    """
    if not HAS_ULTRALYTICS:
        logger.warning("ultralytics not installed. Install with: pip install ultralytics")
        return None
    
    if not HAS_TORCH:
        logger.warning("torch not installed. Install with: pip install torch")
        return None
    
    if not HAS_CV2:
        logger.warning("opencv-python not installed. Install with: pip install opencv-python")
        return None
    
    try:
        # Auto-detect device if not specified
        if device is None:
            if HAS_TORCH:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"  # Apple Silicon
                else:
                    device = "cpu"
        
        # Use default model if not specified
        # Try to use a document layout detection model, fallback to general YOLO
        if model_path is None:
            # Try to load a document layout model (you can replace with your trained model)
            # For now, we'll use a general YOLO model and adapt it
            try:
                # Use YOLOv8n as a base (lightweight)
                model = YOLO("yolov8n.pt")
                logger.info(f"Loaded default YOLO model on {device}")
            except Exception as e:
                logger.warning(f"Could not load default YOLO model: {e}")
                return None
        else:
            model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path} on {device}")
        
        # Set device
        model.to(device)
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None


def detect_layout_elements(
    model: Any,
    image: Any,  # numpy array or PIL Image
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Detect layout elements in a document image using YOLO.
    
    Args:
        model: YOLO model instance
        image: Image as numpy array or PIL Image
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Dictionary with detections: {
            'boxes': List of bounding boxes,
            'scores': List of confidence scores,
            'classes': List of class IDs,
            'class_names': List of class names
        }
    """
    if model is None:
        return {
            'boxes': [],
            'scores': [],
            'classes': [],
            'class_names': []
        }
    
    try:
        # Run inference
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return {
                'boxes': [],
                'scores': [],
                'classes': [],
                'class_names': []
            }
        
        result = results[0]
        
        # Extract detections
        boxes = []
        scores = []
        classes = []
        class_names = []
        
        if hasattr(result, 'boxes'):
            boxes_data = result.boxes
            if boxes_data is not None and len(boxes_data) > 0:
                # Get boxes in xyxy format
                boxes_xyxy = boxes_data.xyxy.cpu().numpy() if hasattr(boxes_data.xyxy, 'cpu') else boxes_data.xyxy
                scores_data = boxes_data.conf.cpu().numpy() if hasattr(boxes_data.conf, 'cpu') else boxes_data.conf
                classes_data = boxes_data.cls.cpu().numpy() if hasattr(boxes_data.cls, 'cpu') else boxes_data.cls
                
                # Get class names
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes_xyxy)):
                    boxes.append(boxes_xyxy[i].tolist())
                    scores.append(float(scores_data[i]))
                    class_id = int(classes_data[i])
                    classes.append(class_id)
                    class_names.append(names.get(class_id, f"class_{class_id}"))
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'class_names': class_names
        }
    
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        return {
            'boxes': [],
            'scores': [],
            'classes': [],
            'class_names': []
        }


def map_yolo_classes_to_layout_types(class_names: list) -> Dict[str, str]:
    """
    Map YOLO class names to document layout types.
    
    This is a mapping function that can be customized based on your trained model.
    
    Args:
        class_names: List of YOLO class names
    
    Returns:
        Dictionary mapping class names to layout types
    """
    # Default mapping (can be customized for specific document layout models)
    layout_mapping = {
        # Common document layout classes
        'title': 'title',
        'heading': 'heading',
        'paragraph': 'paragraph',
        'list': 'list',
        'table': 'table',
        'figure': 'figure',
        'caption': 'caption',
        'equation': 'equation',
        'header': 'header',
        'footer': 'footer',
        'abstract': 'abstract',
        'reference': 'reference',
        # YOLO default classes (if using general model)
        'person': 'figure',  # Might detect figures with people
        'book': 'figure',
        'laptop': 'figure',
    }
    
    # Create mapping for detected classes
    result = {}
    for class_name in class_names:
        class_lower = class_name.lower()
        # Try exact match first
        if class_lower in layout_mapping:
            result[class_name] = layout_mapping[class_lower]
        else:
            # Try partial match
            matched = False
            for key, value in layout_mapping.items():
                if key in class_lower or class_lower in key:
                    result[class_name] = value
                    matched = True
                    break
            if not matched:
                # Default to paragraph
                result[class_name] = 'paragraph'
    
    return result


