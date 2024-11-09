import argparse
import sys
import time
import cv2
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

def draw_text(request: CompletedRequest, text: str, line: int, stream: str = "main"):
    """Draw text onto the ISP output."""
    with MappedArray(request, stream) as m:
        text_left, text_top = 0, 0
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = text_left + 5
        text_y = text_top + 15 + line * 20
        # Create a copy of the array to draw the background with opacity
        overlay = m.array.copy()
        # Draw the background rectangle on the overlay
        cv2.rectangle(overlay,
                      (text_x, text_y - text_height),
                      (text_x + text_width, text_y + baseline),
                      (255, 255, 255),  # Background color (white)
                      cv2.FILLED)
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)
        # Draw text on top of the background
        cv2.putText(m.array, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_result(request: CompletedRequest):
    """Get metadata result from IMX500, interpret it is as an ascii array and show in into camera image"""
    metadata = request.get_metadata()
    np_outputs = imx500.get_outputs(metadata)
    out_array = np_outputs[0]
    text = ''.join(chr(int(i)) for i in out_array)
    draw_text(request, text, 1)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="modelpack/network.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "segmentation"
    elif intrinsics.task != "segmentation":
        print("Network is not a segmentation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    picam2.pre_callback = process_result

    while True:
        time.sleep(0.5)
