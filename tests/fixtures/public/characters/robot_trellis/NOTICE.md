# robot.glb

Generated 2026-11-21T11:31:11+00:00 by img2glb-character, engine `trellis-proj`.

## What you may do with this model

**permissive-with-attribution** — Permissive licences only. Keep NOTICE.md with the model and show the required credits.

Required credit wherever the model or images of it are shown:

- Built with DINOv3

Conditions:

- dinov3: Show "Built with DINOv3" wherever results are shown; follow the acceptable use policy.

## Input pictures

- source: rendered from code in this repo (original_characters/sdf_render.py + scenes/robot.py)
- author: RinoReyns (project owner)
- licence: own work
- notes: No third-party images or models used.
- front.png sha256 02d5a8f612f2b1be763cac33db6346620f2109ca89da71f580bbe2abe9d5c4a1
- left.png sha256 de4f0dda2ac65439d052913dcbd49e89ba200ed1bb7d9c976c1a394fb49eafc6
- back.png sha256 ff2ae33cd1877da85988c4fc60f9969cb52057371068904f6cbb38bbf43e2333
- right.png sha256 8754a566390cf963439af066e0176a2a5349156052cad08d2f3ff4ce3c479c09

## Models and libraries used

| Component | Role | Licence | Version / revision | Link |
|---|---|---|---|---|
| img2glb-character | pipeline (cut-out, clean, place, packaging) | project's own code |  | ~/Repos/image_to_model |
| rembg | background removal library | MIT | 2.0.87 | https://github.com/danielgatis/rembg |
| u2net | background removal weights (u2net.onnx) | Apache-2.0 | sha256 8d10d2f3bb75ae3b | https://github.com/xuebinqin/U-2-Net |
| onnxruntime | runs the u2net weights | MIT | 1.32.0 | https://github.com/microsoft/onnxruntime |
| trimesh | mesh I/O | MIT | 5.2.1 | https://github.com/mikedh/trimesh |
| fast-simplification | mesh reduction | MIT | 0.2.0 | https://github.com/pyvista/fast-simplification |
| numpy | arrays | BSD-3-Clause | 2.4.6 | https://github.com/numpy/numpy |
| scipy | distance fields, filters | BSD-3-Clause | 1.17.1 | https://github.com/scipy/scipy |
| scikit-image | marching cubes, image regions | BSD-3-Clause | 0.26.0 | https://github.com/scikit-image/scikit-image |
| pillow | image I/O | MIT-CMU (HPND) | 13.0.0 | https://github.com/python-pillow/Pillow |
| pytorch | GPU runtime | BSD-3-Clause | 2.7.1+cu128 | https://github.com/pytorch/pytorch |
| trellis2 | shape model (microsoft/TRELLIS.2-4B) + trellis2 code + O-Voxel | MIT | hf revision af44b45f2e35, git 75fbf0183001 | https://github.com/microsoft/TRELLIS.2 |
| trellis-image-large | sparse-structure decoder used by TRELLIS.2 (microsoft/TRELLIS-image-large) | MIT | hf revision 25e0d31ffbeb | https://huggingface.co/microsoft/TRELLIS-image-large |
| dinov3 | image encoder inside TRELLIS.2 (facebook/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License (Meta, custom, worldwide) | hf revision ea8dc2863c51 | https://ai.meta.com/resources/models-and-libraries/dinov3-license |
| cumesh | mesh hole filling (CuMesh core, without its cubvh part) | MIT | 0.0.1 | https://github.com/JeffreyXiang/CuMesh |
| flexgemm | sparse convolution kernels | MIT | 1.0.0 | https://github.com/JeffreyXiang/FlexGEMM |
| xformers | attention kernels | BSD-3-Clause | 0.0.31.post1 | https://github.com/facebookresearch/xformers |
| utils3d | 3D utilities | MIT | 0.0.2 | https://github.com/EasternJournalist/utils3d |
| projection-colours | colours projected from the input pictures (img2glb-character, numpy) | project's own code |  | ~/Repos/image_to_model |

Restricted (NVIDIA research-only) modules loaded during the run: none.

This notice records licences as read from each project; it is not legal advice.
