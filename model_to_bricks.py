"""Command-line entry point, kept so the documented commands keep working:
    python model_to_bricks.py MODEL.glb [options]
The code lives in the image_to_bricks package (image_to_bricks/cli.py)."""

from image_to_bricks.cli import main

if __name__ == "__main__":
    main()
