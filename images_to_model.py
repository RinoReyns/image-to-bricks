"""Command-line entry point: pictures of a character -> coloured 3D model (GLB) with its licence package.
    python images_to_model.py --views DIR --output X.glb --height METRES
The code lives in image_to_bricks/images_to_model (cli.py)."""

from image_to_bricks.images_to_model.cli import main

if __name__ == "__main__":
    main()
