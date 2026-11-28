# Third-party components of the model step (images_to_model, engine trellis-proj)

Everything scripts/setup_images_to_model.sh installs or downloads and the model step runs, with its licence (the
brick step's packages are in requirements.txt). Read from each project's licence
(2026-11-21); this is a record, not legal advice. Every generated model also gets its own NOTICE.md and
PROVENANCE.json with the exact versions used.

**Required credit wherever models made with this project (or images of them) are shown: "Built with DINOv3".**

| Component | Role | Licence | Link |
|---|---|---|---|
| rembg | background removal library | MIT | https://github.com/danielgatis/rembg |
| u2net | background removal weights (u2net.onnx) | Apache-2.0 | https://github.com/xuebinqin/U-2-Net |
| onnxruntime | runs the u2net weights | MIT | https://github.com/microsoft/onnxruntime |
| trimesh | mesh I/O | MIT | https://github.com/mikedh/trimesh |
| fast-simplification | mesh reduction | MIT | https://github.com/pyvista/fast-simplification |
| numpy | arrays | BSD-3-Clause | https://github.com/numpy/numpy |
| scipy | distance fields, filters | BSD-3-Clause | https://github.com/scipy/scipy |
| scikit-image | marching cubes, image regions | BSD-3-Clause | https://github.com/scikit-image/scikit-image |
| pillow | image I/O | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |
| pytorch | GPU runtime | BSD-3-Clause | https://github.com/pytorch/pytorch |
| trellis2 | shape model (microsoft/TRELLIS.2-4B) + trellis2 code + O-Voxel | MIT | https://github.com/microsoft/TRELLIS.2 |
| trellis-image-large | sparse-structure decoder used by TRELLIS.2 (microsoft/TRELLIS-image-large) | MIT | https://huggingface.co/microsoft/TRELLIS-image-large |
| dinov3 | image encoder inside TRELLIS.2 (facebook/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License (Meta, custom, worldwide) | https://ai.meta.com/resources/models-and-libraries/dinov3-license |
| cumesh | mesh hole filling (CuMesh core, without its cubvh part) | MIT | https://github.com/JeffreyXiang/CuMesh |
| flexgemm | sparse convolution kernels | MIT | https://github.com/JeffreyXiang/FlexGEMM |
| xformers | attention kernels | BSD-3-Clause | https://github.com/facebookresearch/xformers |
| utils3d | 3D utilities | MIT | https://github.com/EasternJournalist/utils3d |

Conditions:

- dinov3: Show "Built with DINOv3" wherever results are shown; follow the acceptable use policy.

Also installed: the remaining Python packages in `lock/images_to_model/requirements-gpu.txt` (dependencies of the above;
permissive licences: MIT, BSD, Apache-2.0, PSF, MPL-2.0 for certifi), and NVIDIA CUDA runtime libraries
(`nvidia-*` wheels, NVIDIA CUDA EULA, redistributable runtime parts).

Not installed and not used: nvdiffrast, nvdiffrec (NVIDIA Source Code License, research/evaluation only);
briaai/RMBG-2.0 (CC BY-NC, replaced by the project's u2net cut-out). CuMesh ships cubvh code that contains NVIDIA
instant-ngp parts; it is built but never loaded (checked at every run and recorded in each NOTICE.md).
