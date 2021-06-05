"""
To run it from installed WHL:
$ facetracker

To run it from source tree:
$ python -m openseeface.facetracker
"""

import argparse
import copy
import gc
import os
import signal
import sys
import traceback


class Facetracker(object):
    def __init__(self):
        self._is_running = True
        self._args = self._parse_args()

        signal.signal(signal.SIGHUP, self._handle_signal)  # reload
        signal.signal(signal.SIGTERM, self._handle_signal)  # kill
        signal.signal(signal.SIGINT, self._handle_signal)  # CTRL+C

    def _handle_signal(self, signalnum, frame):
        self._is_running = False

    def _parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
        parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)

        if os.name == 'nt':
            parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
            parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
            parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
            parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
            parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
            parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
            parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
        else:
            parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
            parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)

        parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
        parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
        parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
        parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
        parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
        parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
        parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
        parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
        parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=1)
        parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
        parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
        parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=10)
        parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
        parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
        parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
        parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
        parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
        parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
        parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
        parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
        parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
        parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
        parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
        parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
        parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
        parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
        parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
        parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
        parser.add_argument("--mirror", action="store_true", required=False, help="Mirror the camera image")
        parser.add_argument("--limit-fps", type=int, help="Limit app's max frame rate")
        parser.add_argument("--protocol", type=int, help="Protocol version to use", default=1)
        parser.add_argument("--write-pid", type=str, help="Write process ID to file", required=False)
        parser.add_argument("--hands", type=int, help="Set the maximum number of hands", default=0)
        parser.add_argument("--no-landmarks", action="store_true", required=False, help="Don't send face landmarks")
        parser.add_argument("--no-points", action="store_true", required=False, help="Don't send face points")

        if os.name == 'nt':
            parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
            parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
            parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])

        return parser.parse_args()

    def _write_header(self, log):
        log.write(
            'Frame,Time,Width,Height,FPS,Face,FaceID,RightOpen,LeftOpen,'
            'AverageConfidence,Success3D,PnPError,'
            'RotationQuat.X,RotationQuat.Y,RotationQuat.Z,RotationQuat.W,'
            'Euler.X,Euler.Y,Euler.Z,'
            'RVec.X,RVec.Y,RVec.Z,'
            'TVec.X,TVec.Y,TVec.Z')

        for i in range(66):
            log.write(f',Landmark[{i}].X,Landmark[{i}].Y,Landmark[{i}].Confidence')

        for i in range(66):
            log.write(f',Point3D[{i}].X,Point3D[{i}].Y,Point3D[{i}].Z')

        for feature in const.FEATURES:
            log.write(f',{feature}')

        log.write('\r\n')

    def _set_pixel(self, frame, x, y, color):
        height, width, channels = frame.shape

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x2 = x + dx
                y2 = y + dy
                if (0 <= x2 < height) and (0 <= y2 < width):
                    frame[int(x2), int(y2)] = color

    def run(self):
        if self._args.write_pid:
            if os.path.exists(self._args.write_pid):
                os.remove(self._args.write_pid)

            with open(self._args.write_pid, 'w') as f:
                f.write(str(os.getpid()))

        os.environ["OMP_NUM_THREADS"] = str(self._args.max_threads)

        class OutputLog(object):
            def __init__(self, fh, output):
                self.fh = fh
                self.output = output
            def write(self, buf):
                if not self.fh is None:
                    self.fh.write(buf)
                self.output.write(buf)
                self.flush()
            def flush(self):
                if not self.fh is None:
                    self.fh.flush()
                self.output.flush()
        output_logfile = None
        if self._args.log_output != "":
            output_logfile = open(self._args.log_output, "w")
        sys.stdout = OutputLog(output_logfile, sys.stdout)
        sys.stderr = OutputLog(output_logfile, sys.stderr)

        if os.name == 'nt':
            from . import dshowcapture
            if self._args.blackmagic == 1:
                dshowcapture.set_bm_enabled(True)
            if not self._args.blackmagic_options is None:
                dshowcapture.set_options(self._args.blackmagic_options)
            if not self._args.priority is None:
                import psutil
                classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
                p = psutil.Process(os.getpid())
                p.nice(classes[self._args.priority])

        if os.name == 'nt' and (self._args.list_cameras > 0 or not self._args.list_dcaps is None):
            cap = dshowcapture.DShowCapture()
            info = cap.get_info()
            unit = 10000000.;
            if not self._args.list_dcaps is None:
                formats = {0: "Any", 1: "Unknown", 100: "ARGB", 101: "XRGB", 200: "I420", 201: "NV12", 202: "YV12", 203: "Y800", 300: "YVYU", 301: "YUY2", 302: "UYVY", 303: "HDYC (Unsupported)", 400: "MJPEG", 401: "H264" }
                for cam in info:
                    if self._args.list_dcaps == -1:
                        type_ = ""
                        if cam['type'] == "Blackmagic":
                            type_ = "Blackmagic: "
                        print(f"{cam['index']}: {type_}{cam['name']}")
                    if self._args.list_dcaps != -1 and self._args.list_dcaps != cam['index']:
                        continue
                    for caps in cam['caps']:
                        format = caps['format']
                        if caps['format'] in formats:
                            format = formats[caps['format']]
                        if caps['minCX'] == caps['maxCX'] and caps['minCY'] == caps['maxCY']:
                            print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
                        else:
                            print(f"    {caps['id']}: Resolution: {caps['minCX']}x{caps['minCY']}-{caps['maxCX']}x{caps['maxCY']} FPS: {unit/caps['maxInterval']:.3f}-{unit/caps['minInterval']:.3f} Format: {format}")
            else:
                if self._args.list_cameras == 1:
                    print("Available cameras:")
                for cam in info:
                    type_ = ""
                    if cam['type'] == "Blackmagic":
                        type_ = "Blackmagic: "
                    if self._args.list_cameras == 1:
                        print(f"{cam['index']}: {type_}{cam['name']}")
                    else:
                        print(f"{type_}{cam['name']}")
            cap.destroy_capture()
            sys.exit(0)

        if self._args.hands > 0:
            import mediapipe

        import numpy as np
        import time
        import cv2
        import socket
        import struct
        import json

        from . import const
        from .input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
        from .tracker import Tracker, get_model_base_path


        if self._args.benchmark > 0:
            model_base_path = get_model_base_path(self._args.model_dir)
            im = cv2.imread(os.path.join(model_base_path, "benchmark.bin"), cv2.IMREAD_COLOR)
            results = []
            for model_type in [3, 2, 1, 0, -1, -2, -3]:
                tracker = Tracker(224, 224, threshold=0.1, max_threads=self._args.max_threads, max_faces=1, discard_after=0, scan_every=0, silent=True, model_type=model_type, model_dir=self._args.model_dir, no_gaze=(model_type == -1), detection_threshold=0.1, use_retinaface=0, max_feature_updates=900, static_model=True if self._args.no_3d_adapt == 1 else False)
                tracker.detected = 1
                tracker.faces = [(0, 0, 224, 224)]
                total = 0.0
                for i in range(100):
                    start = time.perf_counter()
                    r = tracker.predict(im)
                    total += time.perf_counter() - start
                print(1. / (total / 100.))
            sys.exit(0)

        socket_af, socket_sk, x, y, socket_addr = socket.getaddrinfo(
            self._args.ip, self._args.port, socket.AF_INET, socket.SOCK_DGRAM)[0]

        if self._args.faces >= 40:
            print("Transmission of tracking data over network is not supported with 40 or more faces.")

        fps = 0
        dcap = None
        use_dshowcapture_flag = False
        if os.name == 'nt':
            fps = self._args.fps
            dcap = self._args.dcap
            use_dshowcapture_flag = True if self._args.use_dshowcapture == 1 else False
            input_reader = InputReader(self._args.capture, self._args.raw_rgb, self._args.width, self._args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap, mirror=mirror)
            if self._args.dcap == -1 and type(input_reader) == DShowCaptureReader:
                fps = min(fps, input_reader.device.get_fps())
        else:
            fps = 0
            input_reader = InputReader(self._args.capture, self._args.raw_rgb, self._args.width, self._args.height, fps, use_dshowcapture=use_dshowcapture_flag, mirror=self._args.mirror)
        if type(input_reader.reader) == VideoReader:
            fps = 0

        log = None
        out = None
        first = True
        height = 0
        width = 0
        tracker = None
        sock = None
        total_tracking_time = 0.0
        tracking_time = 0.0
        tracking_frames = 0
        frame_count = 0
        mp_solver = None

        if self._args.log_data != "":
            log = open(self._args.log_data, "w")
            self._write_header(log)
            log.flush()

        is_camera = self._args.capture == str(try_int(self._args.capture))

        attempt = 0
        frame_time = time.perf_counter()
        target_duration = 0
        if fps > 0:
            target_duration = 1. / float(fps)
        repeat = self._args.repeat_video != 0 and type(input_reader.reader) == VideoReader
        need_reinit = 0
        failures = 0
        source_name = input_reader.name
        while (repeat or input_reader.is_open()) and self._is_running:
            time_ns = time.time_ns()

            if not input_reader.is_open() or need_reinit == 1:
                input_reader = InputReader(self._args.capture, self._args.raw_rgb, self._args.width, self._args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap, mirror=self._args.mirror)
                if input_reader.name != source_name:
                    print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                    sys.exit(1)
                need_reinit = 2
                time.sleep(0.02)
                continue
            if not input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = input_reader.read()
            if ret and self._args.mirror_input:
                frame = cv2.flip(frame, 1)
            if not ret:
                if repeat:
                    if need_reinit == 0:
                        need_reinit = 1
                    continue
                elif is_camera:
                    attempt += 1
                    if attempt > 30:
                        break
                    else:
                        time.sleep(0.02)
                        if attempt == 3:
                            need_reinit = 1
                        continue
                else:
                    break;

            attempt = 0
            need_reinit = 0
            frame_count += 1
            now = time.time()

            if first:
                first = False
                height, width, channels = frame.shape
                sock = socket.socket(socket_af, socket_sk)
                tracker = Tracker(width, height, threshold=self._args.threshold, max_threads=self._args.max_threads, max_faces=self._args.faces, discard_after=self._args.discard_after, scan_every=self._args.scan_every, silent=False if self._args.silent == 0 else True, model_type=self._args.model, model_dir=self._args.model_dir, no_gaze=False if self._args.gaze_tracking != 0 and self._args.model != -1 else True, detection_threshold=self._args.detection_threshold, use_retinaface=self._args.scan_retinaface, max_feature_updates=self._args.max_feature_updates, static_model=True if self._args.no_3d_adapt == 1 else False, try_hard=self._args.try_hard == 1)
                if not self._args.video_out is None:
                    out = cv2.VideoWriter(self._args.video_out, cv2.VideoWriter_fourcc('F','F','V','1'), self._args.video_fps, (width * self._args.video_scale, height * self._args.video_scale))

            try:
                inference_start = time.perf_counter()
                faces = tracker.predict(frame)

                if len(faces) > 0:
                    inference_time = (time.perf_counter() - inference_start)
                    total_tracking_time += inference_time
                    tracking_time += inference_time / len(faces)
                    tracking_frames += 1
                packet = bytearray()
                detected = False

                ###############################################################
                # FACES
                ###############################################################

                if self._args.protocol >= 2:
                    packet.extend(bytearray(struct.pack("B", len(faces))))

                for face_num, f in enumerate(faces):
                    f = copy.copy(f)
                    f.id += self._args.face_id_offset
                    if f.eye_blink is None:
                        f.eye_blink = [1, 1]
                    right_state = "O" if f.eye_blink[0] > 0.30 else "-"
                    left_state = "O" if f.eye_blink[1] > 0.30 else "-"
                    if self._args.silent == 0:
                        print(f"Confidence[{f.id}]: {f.conf:.4f} / 3D fitting error: {f.pnp_error:.4f} / Eyes: {left_state}, {right_state}")
                    detected = True
                    if not f.success:
                        pts_3d = np.zeros((70, 3), np.float32)

                    ###########################################################
                    # FACE DATA
                    ###########################################################

                    packet.extend(bytearray(struct.pack("d", now)))
                    packet.extend(bytearray(struct.pack("i", f.id)))
                    packet.extend(bytearray(struct.pack("f", width)))
                    packet.extend(bytearray(struct.pack("f", height)))
                    packet.extend(bytearray(struct.pack("f", f.eye_blink[0])))
                    packet.extend(bytearray(struct.pack("f", f.eye_blink[1])))
                    packet.extend(bytearray(struct.pack("B", 1 if f.success else 0)))
                    packet.extend(bytearray(struct.pack("f", f.pnp_error)))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[0])))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[1])))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[2])))
                    packet.extend(bytearray(struct.pack("f", f.quaternion[3])))
                    packet.extend(bytearray(struct.pack("f", f.euler[0])))
                    packet.extend(bytearray(struct.pack("f", f.euler[1])))
                    packet.extend(bytearray(struct.pack("f", f.euler[2])))
                    packet.extend(bytearray(struct.pack("f", f.translation[0])))
                    packet.extend(bytearray(struct.pack("f", f.translation[1])))
                    packet.extend(bytearray(struct.pack("f", f.translation[2])))
                    if not log is None:
                        log.write(f"{frame_count},{now},{width},{height},{fps},{face_num},{f.id},{f.eye_blink[0]},{f.eye_blink[1]},{f.conf},{f.success},{f.pnp_error},{f.quaternion[0]},{f.quaternion[1]},{f.quaternion[2]},{f.quaternion[3]},{f.euler[0]},{f.euler[1]},{f.euler[2]},{f.rotation[0]},{f.rotation[1]},{f.rotation[2]},{f.translation[0]},{f.translation[1]},{f.translation[2]}")

                    ###########################################################
                    # LANDMARKS
                    ###########################################################

                    if self._args.protocol >= 2:
                        if not self._args.no_landmarks:
                            count = len(f.lms)
                        else:
                            count = 0
                        packet.extend(bytearray(struct.pack("B", count)))

                    if not self._args.no_landmarks:
                        for (x, y, c) in f.lms:
                            packet.extend(bytearray(struct.pack("f", c)))

                    if self._args.visualize > 1:
                        frame = cv2.putText(frame, str(f.id), (int(f.bbox[0]), int(f.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255))
                    if self._args.visualize > 2:
                        frame = cv2.putText(frame, f"{f.conf:.4f}", (int(f.bbox[0] + 18), int(f.bbox[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

                    if not self._args.no_landmarks:
                        for pt_num, (x, y, c) in enumerate(f.lms):
                            packet.extend(bytearray(struct.pack("f", y)))
                            packet.extend(bytearray(struct.pack("f", x)))
                            if not log is None:
                                log.write(f",{y},{x},{c}")
                            if pt_num == 66 and (f.eye_blink[0] < 0.30 or c < 0.30):
                                continue
                            if pt_num == 67 and (f.eye_blink[1] < 0.30 or c < 0.30):
                                continue
                            if self._args.visualize != 0 or not out is None:
                                if self._args.visualize > 3:
                                    frame = cv2.putText(frame, str(pt_num), (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,0))
                                if pt_num >= 66:
                                    color = const.COLOR_RED
                                else:
                                    color = const.COLOR_GREEN
                                self._set_pixel(frame, x, y, color)

                        if self._args.pnp_points != 0 and (self._args.visualize != 0 or not out is None) and f.rotation is not None:
                            if self._args.pnp_points > 1:
                                projected = cv2.projectPoints(f.face_3d[0:66], f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                            else:
                                projected = cv2.projectPoints(f.contour, f.rotation, f.translation, tracker.camera, tracker.dist_coeffs)
                            for [(x, y)] in projected[0]:
                                self._set_pixel(frame, x, y, const.COLOR_YELLOW)

                    ###########################################################
                    # POINTS
                    ###########################################################

                    if self._args.protocol >= 2:
                        if not self._args.no_points:
                            count = len(f.pts_3d)
                        else:
                            count = 0
                        packet.extend(bytearray(struct.pack("B", count)))

                    if not self._args.no_points:
                        for (x, y, z) in f.pts_3d:
                            packet.extend(bytearray(struct.pack("f", x)))
                            packet.extend(bytearray(struct.pack("f", -y)))
                            packet.extend(bytearray(struct.pack("f", -z)))
                            if not log is None:
                                log.write(f",{x},{-y},{-z}")

                    ###########################################################
                    # FEATURES
                    ###########################################################

                    if f.current_features is None:
                        f.current_features = {}
                    for feature in const.FEATURES:
                        if not feature in f.current_features:
                            f.current_features[feature] = 0
                        packet.extend(bytearray(struct.pack("f", f.current_features[feature])))
                        if not log is None:
                            log.write(f",{f.current_features[feature]}")
                    if not log is None:
                        log.write("\r\n")
                        log.flush()

                ###############################################################
                # HANDS
                ###############################################################

                if self._args.protocol >= 2:  # hands for protocol v2+ only
                    hand_landmarks = []

                    if self._args.hands > 0:
                        if mp_solver is None:
                            mp_solver = mediapipe.solutions.hands.Hands(
                                max_num_hands=self._args.hands,
                                static_image_mode=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

                        mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_solution = mp_solver.process(mp_frame)
                        if mp_solution.multi_handedness and mp_solution.multi_hand_landmarks:
                            for i, (hand, landmarks) in enumerate(zip(
                                    mp_solution.multi_handedness,
                                    mp_solution.multi_hand_landmarks)):
                                c = hand.classification[0]

                                is_left = c.label == 'Right'  # mediapipe expects mirrored image
                                if self._args.mirror:
                                    is_left = not is_left

                                hand_landmarks.append((is_left, []))
                                for j, landmark in enumerate(landmarks.landmark):
                                    hand_landmarks[-1][1].append(landmark)
                                    if is_left:
                                        color = const.COLOR_RED
                                    else:
                                        color = const.COLOR_CYAN
                                    self._set_pixel(frame, landmark.y * height, landmark.x * width, color)

                                if i >= self._args.hands - 1:
                                    break

                    packet.extend(bytearray(struct.pack("B", len(hand_landmarks))))
                    for is_left, landmarks in hand_landmarks:
                        packet.extend(bytearray(struct.pack("B", 1 if is_left else 0)))
                        packet.extend(bytearray(struct.pack("B", len(landmarks))))
                        for landmark in landmarks:
                            packet.extend(bytearray(struct.pack("f", landmark.x)))
                            packet.extend(bytearray(struct.pack("f", landmark.y)))
                            packet.extend(bytearray(struct.pack("f", landmark.z)))

                if detected and len(faces) < 40:
                    if self._args.protocol >= 2:
                        checksum = sum(packet) & 0xffff  # to uint16
                        header = bytearray(struct.pack('H', checksum))
                        packet = header + packet
                    sock.sendto(packet, socket_addr)

                if not out is None:
                    video_frame = frame
                    if self._args.video_scale != 1:
                        video_frame = cv2.resize(frame, (width * self._args.video_scale, height * self._args.video_scale), interpolation=cv2.INTER_NEAREST)
                    out.write(video_frame)
                    if self._args.video_scale != 1:
                        del video_frame

                if self._args.visualize != 0:
                    cv2.imshow('OpenSeeFace Visualization', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        if self._args.dump_points != "" and not faces is None and len(faces) > 0:
                            np.set_printoptions(threshold=sys.maxsize, precision=15)
                            points = copy.copy(faces[0].face_3d)
                            for a, b in const.POINTS_PAIRS:
                                x = (points[a, 0] - points[b, 0]) / 2.0
                                y = (points[a, 1] + points[b, 1]) / 2.0
                                z = (points[a, 2] + points[b, 2]) / 2.0
                                points[a, 0] = x
                                points[b, 0] = -x
                                points[[a, b], 1] = y
                                points[[a, b], 2] = z
                            points[[8, 27, 28, 29, 33, 50, 55, 60, 64], 0] = 0.0
                            points[30, :] = 0.0
                            with open(self._args.dump_points, "w") as fh:
                                fh.write(repr(points))
                        break
                failures = 0
            except Exception as e:
                if e.__class__ == KeyboardInterrupt:
                    if self._args.silent == 0:
                        print("Quitting")
                    break
                traceback.print_exc()
                failures += 1
                if failures > 30:
                    break

            collected = False
            del frame

            duration = time.perf_counter() - frame_time
            while duration < target_duration:
                if not collected:
                    gc.collect()
                    collected = True
                duration = time.perf_counter() - frame_time
                sleep_time = target_duration - duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
                duration = time.perf_counter() - frame_time
            frame_time = time.perf_counter()

            if self._args.limit_fps:
                dt = time.time_ns() - time_ns
                target_ms = 1000 / self._args.limit_fps
                wait = max(0, (target_ms * 1000000 - dt) / 1000000)
                time.sleep(wait / 1000)

        if self._args.silent == 0:
            print("Quitting")

        input_reader.close()
        if not out is None:
            out.release()
        cv2.destroyAllWindows()

        if self._args.silent == 0 and tracking_frames > 0:
            average_tracking_time = 1000 * tracking_time / tracking_frames
            print(f"Average tracking time per detected face: {average_tracking_time:.2f} ms")
            print(f"Tracking time: {total_tracking_time:.3f} s\nFrames: {tracking_frames}")


def main():
    ft = Facetracker()
    ft.run()


if __name__ == '__main__':
    main()
