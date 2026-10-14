# robot-grasping

Two-finger grasp prediction from a single RGB image. The pipeline segments the object, recovers its 6-DoF pose and a 3D mesh, scores every surface point for grasp stability, and projects ranked two-finger grasp rectangles back onto the image.

## How it works

```
RGB + depth ──► SAM 2 mask ──► Any6D pose + scale ──► stability heatmap ──► grasp candidates
                                   ▲
                    InstantMesh mesh (optional, image-to-3D)
```

1. **Segmentation (SAM 2).** `run_sam.py` generates automatic masks; the target object's mask is saved as `labels_filtered.npz`.
2. **Mesh (InstantMesh, optional).** With `--img_to_3d`, the masked crop is lifted to a textured mesh, so no CAD model is needed.
3. **Pose (Any6D / FoundationPose).** `run_demo_optimised.py` registers the mesh to the image, estimating 6-DoF pose and metric scale. Inputs are downscaled (default 320 px, intrinsics rescaled) so it fits on a T4 GPU. `depth.png` is treated as a disparity map, clipped to the 1st–99th percentile and inverted to relative depth.
4. **Stability heatmap.** `get_heatmap.py` scores each mesh vertex using surface normal vs. gravity (top / side / bottom faces), distance from the centre of gravity, height and distance from the edge, then projects the scores into the image and blurs them into a heatmap.
5. **Grasp candidates.** From the top-scoring points, the second finger contact is found either by ray casting across the 2D mask (default) or by finding the opposite mesh vertex (`--use_vertex`). Each candidate is drawn as a scored grasp rectangle.

A separate training-free 2D baseline lives in `Grasp_Detection_AppWorking_Method_With_Cropping.ipynb`: Depth Anything V2 Small depth, edge + depth-gradient + CoG heuristics, evaluated on the Cornell Grasping Dataset with IoU / angle metrics.

## Repo layout

| Path | Contents |
|---|---|
| `src/run_sam.py` | SAM 2 automatic mask generation |
| `src/run_demo_optimised.py` | Mesh + 6-DoF pose estimation |
| `src/estimater.py` | `Any6D` estimator class |
| `src/get_heatmap.py` | Stability heatmap and grasp candidates (CLI) |
| `src/*.ipynb` | Colab setup for Any6D and InstantMesh |
| `demo/` | Flask web app for interactive heatmap tuning |
| `results/` | Example outputs for `red_bottle` and `cornell1` |

## Setup

The pose stage runs inside an [Any6D](https://github.com/taeyeopl/Any6D) checkout (it imports `foundationpose`, `sam2_instantmesh` and `nvdiffrast`). `src/Any6D_main.ipynb` walks through the full Colab install: Any6D, nvdiffrast, Kaolin, PyTorch3D, SAM 2, InstantMesh and bop_toolkit. Copy the files in `src/` into the Any6D root.

SAM 2 expects `sam2/checkpoints/sam2.1_hiera_large.pt`.

## Usage

Put each object's inputs in `demo_data_<obj>/`:

```
color.png                  RGB image
depth.png                  disparity / relative depth
labels_filtered.npz        SAM 2 mask for the object
836212060125_640x480.yml   camera intrinsics
<obj>.obj                  mesh (only if not using --img_to_3d)
```

```bash
# 1. Pose (+ mesh from the image)
python run_demo_optimised.py --obj red_bottle --img_to_3d --max_size 320

# 2. Heatmap and grasps (reads results/demo_<obj>/)
python get_heatmap.py --obj red_bottle            # mask ray casting
python get_heatmap.py --obj red_bottle --use_vertex --use_cog_avg
```

Pose outputs go to `results/demo_<obj>/` (`<obj>_poses.npy`, `<obj>_scores.npy`, `<obj>_mesh_scale.npy`, `final_mesh_<obj>.obj`). Heatmaps and grasp images go to `stability_results_<obj>/`.

### Web demo

```bash
cd demo
pip install -r requirements.txt
python server.py          # http://localhost:5000
```

Upload the mesh, color image, mask, poses, intrinsics (plus optional scores, mesh scale and resize metadata) from a results folder, then adjust the face thresholds, scoring weights and heatmap blur and regenerate the heatmap and grasps live. The example folders in `results/` work as-is.

## Built on

[Any6D](https://github.com/taeyeopl/Any6D) · [FoundationPose](https://github.com/NVlabs/FoundationPose) · [SAM 2](https://github.com/facebookresearch/sam2) · [InstantMesh](https://github.com/TencentARC/InstantMesh) · [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
