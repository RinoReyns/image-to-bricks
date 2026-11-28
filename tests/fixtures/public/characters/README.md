# Test characters (robot, frog, bear)

Made from scratch with code (`sdf_render.py` + `scenes/*.py`); no third-party images or models, so they are own work
(see each `SOURCE.json`). Each folder holds 512 x 512 RGBA pictures with a transparent background: front, left, back,
right ("left" = taken from the character's left side).

- Input of the model step tests: `python images_to_bricks.py --views tests/fixtures/public/characters/robot ...`
- True shapes for `scripts/reproduce_images_to_model.sh`: `python -m tools.images_to_model.build_truth
  tests/fixtures/public/characters --out repro_out/truth` (marching cubes of the same distance functions)
- `robot_trellis/robot.glb`: the model TRELLIS.2 (engine trellis-proj) made from `robot/` (with its `NOTICE.md`,
  `PROVENANCE.json`): a real model-step output for the brick-step tests without a GPU. Built with DINOv3.

Re-render (e.g. at 1024 px; the committed pictures are 512 px):

    python tests/fixtures/public/characters/sdf_render.py tests/fixtures/public/characters/scenes/robot.py robot 1024
