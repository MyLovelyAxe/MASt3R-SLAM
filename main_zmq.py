import os
import zmq
import cv2
import time
import torch
import pickle
import argparse
import datetime
import lietorch
import numpy as np
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R

from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)


### fixed values
IMG_RESIZE_WIDTH = 512
IMG_RESIZE_HEIGHT = 384

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


def extract_image(
    parts: list, 
    id:int, 
    save: bool = False,
) -> np.ndarray:
    """
    Extracts the image from the received parts.
    """
    img_format = parts[2].decode()
    img_data = parts[3]
    # print(f"Received image in format: {img_format}, data length: {len(img_data)} bytes")
    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if save:
        save_path = 'datasets/tmp'
        os.makedirs(save_path, exist_ok=True)
        img_name = f"image_{id}.{img_format}"
        with open(os.path.join(save_path, img_name), 'wb') as f:
            f.write(img_data)
    return img


def create_world_rotation() -> np.ndarray:
    """ Create a rotation matrix that rotates the world coordinate system to match the OpenGL coordinate system."""
    world_rot = np.eye(4)
    rot_y90 = R.from_euler('y', np.deg2rad(90)).as_matrix()
    rot_x90 = R.from_euler('x', np.deg2rad(90)).as_matrix()
    world_rot[:3, :3] = rot_y90 @ rot_x90
    return world_rot


def rectify_orientation(
    world_rot: np.ndarray,
    points: np.ndarray, # shape: [N, 3]
) -> np.ndarray:
    """
    Rectify the orientation of the point clouds to match the world coordinate system.
    """
    # convert to homogeneous coordinates
    points = np.concatenate([
        points, 
        np.full(points.shape[0], 1, dtype=np.float32).reshape(-1, 1) # shape: [N,1]
    ], axis=1) # shape: [N,4]
    # rotate
    points = (points @ (world_rot))
    return points[:, :3]


def rt2hom(
    translation: np.ndarray, 
    rotation: np.ndarray,
) -> np.ndarray:
    """
    Convert translation and rotation to a homogeneous transformation matrix.
    """
    translation = translation.reshape(3)
    homogeneous = np.eye(4)
    homogeneous[:3, :3] = rotation
    homogeneous[:3, 3] = translation
    return homogeneous

def hom2rt(
    homogeneous: np.ndarray,
) -> tuple:
    """
    Convert a homogeneous transformation matrix to translation and rotation.
    """
    translation = homogeneous[:3, 3].reshape(1,3)
    rotation = homogeneous[:3, :3]
    return rotation, translation


if __name__ == "__main__":

    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--dwsp_size", type=int, default=10, help="Downsample size publishing result point cloud")
    parser.add_argument("--calib")
    args = parser.parse_args()

    load_config(args.config)
    print(config)

    manager = mp.Manager()
    model = load_mast3r(device=device)
    model.share_memory()

    ### for receiving compressed images via zmq
    img_context = zmq.Context()
    img_socket = img_context.socket(zmq.SUB)
    img_socket.connect("tcp://localhost:5556")
    img_socket.setsockopt(zmq.SUBSCRIBE, b"") # Subscribe to all topics

    ### for sending out results via zmq
    res_context = zmq.Context()
    res_socket = res_context.socket(zmq.PUB)
    res_socket.bind("tcp://127.0.0.1:5555")

    ### prepare
    # this is the desized size of all imput image, 
    # i.e. all input images should be resized into 512x384, hard-coded
    h, w = IMG_RESIZE_HEIGHT, IMG_RESIZE_WIDTH 
    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)
    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, None))
    backend.start()

    i = 0 # frame index
    fps_timer = time.time()
    frames = []
    world_rot = create_world_rotation()

    while True:

        mode = states.get_mode()

        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        # for live-stream task, only run 300 frames
        if i == 300: 
            states.set_mode(Mode.TERMINATED)
            break

        # receive image
        img_parts = img_socket.recv_multipart()
        if len(img_parts) != 4:
            print(f"Received unexpected number of parts: {len(img_parts)}")
            continue
        img = extract_image(parts=img_parts,id=i)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=IMG_RESIZE_WIDTH, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")


        ### Send results to zmq socket
        if add_new_kf:
            camera_poses = []
            pointclouds = []
            colors = []
            # important:
            #   use "for frame_id in range(len(keyframes))",
            #   do not use "for keyframe in keyframes",
            #   because SharedKeyframes instance was given a fixed length 512 when created
            #   and the __len__ of SharedKeyframes is re-written to return number of real keyframes
            #   check definition of SharedKeyframes
            for frame_id in range(len(keyframes)):

                keyframe = keyframes[frame_id]

                # camera poses
                T_WC = as_SE3(keyframe.T_WC)
                x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
                # note:
                #    here construct vertical stack of translation and rotation
                #    in order to convert to a 4x3 matrix
                #    then it can be combined with point cloud as one array / message
                #    and send to one single zmq socket
                tran = np.array([x, y, z]).reshape(1, 3) # shape: [1,3]
                rot = R.from_quat([qx, qy, qz, qw]).as_matrix() # shape: [3,3]
                # Attention:
                #    the tran and rot are world2cam
                #    so we need to invert the transformation to get cam2world firstly
                #    then convert them back again to world2cam
                hom = rt2hom(translation=tran, rotation=rot)
                hom = np.linalg.inv(np.linalg.inv(hom) @ world_rot)
                rot, tran = hom2rt(hom)
                # prepare output structure
                rot_tran_vtack = np.vstack([tran, rot]) # shape: [4,3]
                camera_poses.append(rot_tran_vtack)
                print(f"Camera {frame_id}: Translation={tran}, Rotation={rot}")

                # point cloud
                pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)[::args.dwsp_size]
                color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)[::args.dwsp_size]
                valid = (
                    keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
                    > last_msg.C_conf_threshold
                )[::args.dwsp_size]
                pointclouds.append(pW[valid])
                colors.append(color[valid])
            pointclouds = np.concatenate(pointclouds, axis=0)
            # rotate point cloud to match world coordinate system
            pointclouds = rectify_orientation(
                world_rot=world_rot,
                points=pointclouds,
            )
            colors = np.concatenate(colors, axis=0)

            # construct the message to send
            numbers = np.array([
                len(pointclouds),
                len(colors),
                len(camera_poses),
            ]).reshape(1,3)
            lst = [numbers, pointclouds, colors] + camera_poses
            res_msg = np.concatenate(lst, axis=0) # shape: [3+num_pcd+num_col+num_cameras*4, 3]
            del lst

            # Serialize and send the array
            msg = pickle.dumps(res_msg)
            res_socket.send(msg)
            print(f"Sent message: ")
            print(f"  - point cloud positions: {pointclouds.shape}")
            print(f"  - point cloud colors: {colors.shape}")
            print(f"  - camera poses: {len(camera_poses)} cameras, each with shape {rot_tran_vtack.shape}")

        i += 1
        print(f'Processed frame {i}')

    print("done")
    backend.join()
