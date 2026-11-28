"""images_to_model step 5: the model is put on the floor, centred and scaled to the asked height."""

import numpy as np
import support
from images_to_model_helpers import assert_raises, build_cube, is_close

from image_to_bricks.images_to_model.mesh import Mesh
from image_to_bricks.images_to_model.place import apply_placement, compute_placement, measure_mesh


def test_placement_puts_the_feet_on_the_ground_centred_and_scaled() -> None:
    cube = build_cube()
    stretched = cube.vertices * np.array([2.0, 4.0, 1.0]) + np.array([5.0, -3.0, 7.0])
    scale, offset = compute_placement(stretched, 1.8)
    numbers = measure_mesh(Mesh(apply_placement(stretched, scale, offset), cube.faces))
    assert is_close(scale, 0.45)
    assert is_close(numbers["min_y"], 0.0) and is_close(numbers["height"], 1.8)
    assert is_close(numbers["width"], 0.9) and is_close(numbers["depth"], 0.45)
    assert is_close(numbers["centre_x"], 0.0) and is_close(numbers["centre_z"], 0.0)
    assert is_close(numbers["volume"], 0.9 * 1.8 * 0.45, rel=1e-5)


def test_placement_rejects_a_bad_height_and_a_flat_mesh() -> None:
    with assert_raises(ValueError, "height must be positive"):
        compute_placement(build_cube().vertices, 0)
    with assert_raises(ValueError, "no height"):
        compute_placement(np.zeros((3, 3)), 1.0)


if __name__ == "__main__":
    support.run_tests(globals())
