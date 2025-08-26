import platform

import cv2


def get_available_cameras() -> tuple[list[int], list[str]]:
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph

            graph = FilterGraph()
            devices = graph.get_input_devices()

            # Create list of indices and names
            camera_indices = list(range(len(devices)))
            camera_names = devices

            # If no cameras found through DirectShow, try OpenCV fallback
            if not camera_names:
                # Try to open camera with index -1 and 0
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            # If still no cameras found, return empty lists
            if not camera_names:
                return [], ["No cameras found"]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":  # macOS specific handling
            # Try to open the default FaceTime camera first
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                camera_indices.append(0)
                camera_names.append("FaceTime Camera")
                cap.release()

            # On macOS, additional cameras typically use indices 1 and 2
            for i in [1, 2]:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names
