import argparse
import sys
import time
import cv2
import numpy as np
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

def show_output(request, img, stream="main"):
  with MappedArray(request, stream) as m:
    #print(img); print("\n")
    # normalize 'tensor' coming from IMX500 to range 0..1
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # transform it to integer in the range 0..255
    img = np.uint8(255 * img)
    # camera stream has resolution 640 x 480, the tensor coming from IMX500 is 224 x 224
    # assign incoming IMX500 RGB planes to corresponding planes in left upper corner of camera stream
    m.array[0:224, 0:224, 0] = img
    m.array[0:224, 0:224, 1] = img
    m.array[0:224, 0:224, 2] = img

def process_result(request: CompletedRequest):
    """Get metadata result from IMX500, interpret it as image and put it into left upper corner of camera stream"""
    metadata = request.get_metadata()
    np_outputs = imx500.get_outputs(metadata)
    out_array = np_outputs[0]
    show_output(request, out_array)

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
