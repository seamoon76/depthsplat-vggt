import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import read_write_model as rw  # COLMAP Python IO

def build_world_to_cam(qvec, tvec):
    R_mat = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    t = np.array(tvec).reshape(3, 1)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t.flatten()
    return T


def normalize_model(input_dir, output_dir):
    cameras = rw.read_cameras_binary(os.path.join(input_dir, "cameras.bin"))
    images = rw.read_images_binary(os.path.join(input_dir, "images.bin"))
    points3D = rw.read_points3D_binary(os.path.join(input_dir, "points3D.bin"))

    ref_img_id = min(images.keys())
    ref_img = images[ref_img_id]
    T_ref = build_world_to_cam(ref_img.qvec, ref_img.tvec)  # w2c
    T_ref_inv = np.linalg.inv(T_ref)  # c2w

    new_images = {}
    for img_id, img in images.items():
        T_img = build_world_to_cam(img.qvec, img.tvec)  # w2c
        T_new = T_ref_inv @ T_img

        R_new = T_new[:3, :3]
        t_new = T_new[:3, 3]
        q_new = R.from_matrix(R_new).as_quat()  # [x, y, z, w]
        qvec_new = np.array([q_new[3], q_new[0], q_new[1], q_new[2]])  # wxyz

        new_img = rw.Image(
            id=img.id,
            qvec=qvec_new,
            tvec=t_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids
        )
        new_images[img_id] = new_img

    new_points3D = {}
    for pid, pt in points3D.items():
        xyz_h = np.ones(4)
        xyz_h[:3] = pt.xyz
        xyz_new = (T_ref_inv @ xyz_h)[:3]

        new_pt = rw.Point3D(
            id=pt.id,
            xyz=xyz_new,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=pt.image_ids,
            point2D_idxs=pt.point2D_idxs
        )
        new_points3D[pid] = new_pt

    os.makedirs(output_dir, exist_ok=True)
    rw.write_cameras_binary(cameras, os.path.join(output_dir, "cameras.bin"))
    rw.write_images_binary(new_images, os.path.join(output_dir, "images.bin"))
    rw.write_points3D_binary(new_points3D, os.path.join(output_dir, "points3D.bin"))

    print(f"Normalized model saved to: {output_dir}")
    print(f"First image ID: {ref_img_id}, name: {ref_img.name}")

def batch_normalize(base_dir):
    scene_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d))]

    for scene_dir in scene_dirs:
        sparse_input = os.path.join(scene_dir, 'sparse')
        sparse_output = os.path.join(scene_dir, 'sparse_norm')
        
        if not os.path.exists(sparse_input):
            print(f"Skipped {scene_dir}: no sparse directory found.")
            continue
        
        print(f"Normalizing scene: {scene_dir}")
        normalize_model(sparse_input, sparse_output)
        
if __name__ == "__main__":
    base_dir = "/home/jiaysun/re10k_vggsfm"
    batch_normalize(base_dir)
