import cv2


def pump_debug_windows() -> None:
    poll_key = getattr(cv2, "pollKey", None)
    if callable(poll_key):
        poll_key()


def close_debug_windows() -> None:
    cv2.destroyAllWindows()
