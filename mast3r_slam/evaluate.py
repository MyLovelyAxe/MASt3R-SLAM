import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement
import trimesh
from scipy.spatial.transform import Rotation as R


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)


######################################
###### Store result in glb file ######
######################################


def pose_to_matrix(x, y, z, qx, qy, qz, qw):
    """Convert pose to 4x4 transformation matrix."""
    rot = R.from_quat([qx, qy, qz, qw])
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = [x, y, z]
    return mat


def create_camera_marker(
    style:str = "cone",
):
    """Create a simple camera frustum or axis mesh."""
    # We'll use a simple axis frame to represent the camera
    if style == "axis":
        cam_marker = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
    elif style == "cone":
        # cam_marker = trimesh.creation.cone(radius=0.08, height=0.06, sections=4)

        width, height, depth = 0.16, 0.12, 0.08
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Camera center at origin
        tip = np.array([0, 0, 0])

        # Image plane corners in camera's local frame (+Z forward)
        hw, hh = width / 2, height / 2
        p0 = np.array([-hw, -hh, depth])  # bottom-left
        p1 = np.array([ hw, -hh, depth])  # bottom-right
        p2 = np.array([ hw,  hh, depth])  # top-right
        p3 = np.array([-hw,  hh, depth])  # top-left

        # Define edges: 4 sides + 4 rectangle edges
        edges = np.array([
            [tip, p0], [tip, p1], [tip, p2], [tip, p3],  # sides
            [p0, p1], [p1, p2], [p2, p3], [p3, p0]       # base rectangle
        ])

        # Convert to Path3D
        cam_marker = trimesh.load_path(edges)
        cam_marker.colors = np.tile(np.append(color, 255), (len(cam_marker.entities), 1))

    return cam_marker


def save_glb(output_file, points, colors, camera_poses):

    if points.shape != colors.shape or points.shape[1] != 3:
        raise ValueError("Points and colors must be of shape (N, 3)")
    if camera_poses.shape[1] != 7:
        raise ValueError("Camera poses must be shape (M, 7)")

    # Create point cloud
    cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

    # Create scene and add cloud
    scene = trimesh.Scene()
    scene.add_geometry(cloud, node_name="point_cloud")

    # Add cameras
    for i, pose in enumerate(camera_poses):
        T = pose_to_matrix(*pose)
        cam_marker = create_camera_marker(style="cone")
        scene.add_geometry(cam_marker, node_name=f"camera_{i}", transform=T)

    # Export
    scene.export(output_file)
    print(f"Saved GLB to: {output_file}")


def save_reconstruction_glb(
    savedir,
    filename,
    keyframes,
    c_conf_threshold,
    intrinsics: Optional[Intrinsics] = None,
):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    ### camera poses
    camera_poses = []
    # for keyframe_id in frames.keyframe_ids:
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if intrinsics is None:
            T_WC = as_SE3(keyframe.T_WC)
        else:
            T_WC = intrinsics.refine_pose_with_calibration(keyframe)
        x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
        camera_poses.append([x, y, z, qx, qy, qz, qw])
    camera_poses = np.array(camera_poses)

    ### point cloud

    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_glb(savedir / filename, pointclouds, colors, camera_poses)