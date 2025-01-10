from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
import threading
import queue
import numpy as np
from flask import Flask, Response

# Disable gradient computation for faster inference
torch.set_grad_enabled(False)

# Initialize Flask application
app = Flask(__name__)

# Global variables to store the latest processed frame
output_frame = None
frame_lock = threading.Lock()

def frame_capturer(vs, frame_queue):
    """
    Capture video frames and put them into a queue.
    """
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)

def generate_frames():
    """
    Generate frames for Flask streaming.
    """
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame = buffer.tobytes()
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    """
    Video streaming route at root URL.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Demo with Optimization and Streaming',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Input arguments
    parser.add_argument('--input', type=str, default='0',
                        help='USB Camera ID, IP Camera URL, or Image Directory/Video File Path')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output frames (if None, do not save)')
    parser.add_argument('--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
                        help='Image matching patterns if input is an image directory')
    parser.add_argument('--skip', type=int, default=1,
                        help='Number of images to skip if input is a video or directory')
    parser.add_argument('--max_length', type=int, default=1000000,
                        help='Maximum length to process if input is a video or directory')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480],
                        help='Resize input images before inference. If two numbers, resize to exact size; if one number, resize the largest dimension; if -1, do not resize')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='indoor',
                        help='SuperGlue model weights selection')
    parser.add_argument('--max_keypoints', type=int, default=200,  # Reduce number of keypoints
                        help='Maximum number of keypoints detected by SuperPoint (use -1 to keep all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.01,  # Increase threshold
                        help='Confidence threshold for SuperPoint keypoint detection')
    parser.add_argument('--nms_radius', type=int, default=4,
                        help='SuperPoint Non-Maximum Suppression (NMS) radius (must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                        help='Number of Sinkhorn iterations for SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2,
                        help='Match threshold for SuperGlue')
    parser.add_argument('--show_keypoints', action='store_true',
                        help='Display detected keypoints')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images on screen. Suitable for remote runs')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force PyTorch to use CPU mode')
    parser.add_argument('--process_every_n_frames', type=int, default=1,
                        help='Process every N frames to reduce computational load')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address for Flask server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port number for Flask server (default: 8080)')
    
    # Parse arguments
    opt = parser.parse_args()
    print(opt)
    
    # Handle resize parameters
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Resizing images to {}x{} (widthxheight)'.format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Resizing images with the largest dimension {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('No resizing of images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not opt.force_cpu else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('Running inference on device "{}"'.format(device))
    
    # Configure SuperPoint and SuperGlue
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    
    # Initialize matching model
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    
    # Initialize video stream
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error reading the first frame (try a different --input?)'
    
    # Convert frame to tensor
    frame_tensor = frame2tensor(frame, device)
    
    # Detect keypoints using SuperPoint
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0
    
    # Create output directory if specified
    if opt.output_dir is not None:
        print('==> Writing output to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)
    
    # Create frame queue and capturer thread
    frame_queue = queue.Queue(maxsize=2)
    capturer_thread = threading.Thread(target=frame_capturer, args=(vs, frame_queue), daemon=True)
    capturer_thread.start()
    
    # Configure Matplotlib (keep comments in Chinese, display text in English)
    plt.rcParams['font.sans-serif'] = ['STHeiti']  # Use macOS built-in Chinese font STHeiti
    plt.rcParams['axes.unicode_minus'] = False    # Fix negative sign display issue
    
    # Create display window
    if not opt.no_display:
        cv2.namedWindow('SuperGlue Matches', cv2.WINDOW_NORMAL)
        # Set window size to match the generated match plot to avoid stretching
        cv2.resizeWindow('SuperGlue Matches', 1280, 960)  # Adjust window size
    else:
        print('Skipping visualization, no GUI will be displayed.')
    
    # Print keyboard control menu
    print('==> Keyboard Control Menu:\n'
          '\tn: Set the current frame as the anchor\n'
          '\te/r: Increase/Decrease keypoint confidence threshold\n'
          '\td/f: Increase/Decrease match filtering threshold\n'
          '\tk: Toggle keypoint visualization\n'
          '\tq: Quit the program')
    
    # Start Flask server thread
    def run_flask():
        app.run(host=opt.host, port=opt.port, debug=False, threaded=True, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Initialize timer and frame counter
    timer = AverageTimer()
    frame_count = 0
    
    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                print('Demo finished.')
                break
            frame_count += 1
    
            # Reduce processing frame rate
            if frame_count % opt.process_every_n_frames != 0:
                continue
    
            # Horizontally flip the captured frame
            frame = cv2.flip(frame, 1)  # 1 means horizontal flip
    
            timer.update('data')
            stem0, stem1 = last_image_id, vs.i - 1
    
            # Convert frame to tensor
            frame_tensor = frame2tensor(frame, device)
    
            # Perform matching using SuperGlue
            pred = matching({**last_data, 'image1': frame_tensor})
            kpts0 = last_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            timer.update('forward')
    
            # Filter valid matches
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            color = cm.jet(confidence[valid])
    
            # Prepare display text (remove all text overlays)
            text = []
            small_text = []
    
            # Generate match plot without any text
            out = make_matching_plot_fast(
                last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
    
            # Update the global output frame for streaming
            with frame_lock:
                output_frame = out.copy()
    
            # Display the match results
            if not opt.no_display:
                cv2.imshow('SuperGlue Matches', out)
                key = cv2.waitKey(1)
                if key == ord('q'):  # Press 'q' to quit
                    print('Exiting demo_superglue_optimized.py by pressing q')
                    break
                elif key == ord('n'):  # Set current frame as anchor
                    last_data = {k+'0': pred[k+'1'] for k in keys}
                    last_data['image0'] = frame_tensor
                    last_frame = frame
                    last_image_id = (vs.i - 1)
                elif key in [ord('e'), ord('r')]:
                    # Increase/Decrease keypoint threshold by 10%
                    d = 0.1 * (-1 if key == ord('e') else 1)
                    matching.superpoint.config['keypoint_threshold'] = min(max(
                        0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                    print('\nKeypoint Threshold changed to {:.4f}'.format(
                        matching.superpoint.config['keypoint_threshold']))
                elif key in [ord('d'), ord('f')]:
                    # Increase/Decrease match threshold by 0.05
                    d = 0.05 * (-1 if key == ord('d') else 1)
                    matching.superglue.config['match_threshold'] = min(max(
                        0.05, matching.superglue.config['match_threshold']+d), .95)
                    print('\nMatch Threshold changed to {:.2f}'.format(
                        matching.superglue.config['match_threshold']))
                elif key == ord('k'):
                    # Toggle keypoint visualization
                    opt.show_keypoints = not opt.show_keypoints
    
            timer.update('viz')
            timer.print()
    
            # Save output image if output directory is specified
            if opt.output_dir is not None:
                stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
                out_file = str(Path(opt.output_dir, stem + '.png'))
                print('\nWriting image to {}'.format(out_file))
                cv2.imwrite(out_file, out)
    
    except KeyboardInterrupt:
        # Handle Ctrl+C exit
        print('\nExiting demo_superglue_optimized.py by pressing Ctrl+C')
    
    finally:
        # Close all windows and release resources
        cv2.destroyAllWindows()
        vs.cleanup()