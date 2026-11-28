"""Command-line entry point: pictures of a character -> 3D model -> LEGO build with instructions.
    python images_to_bricks.py --views DIR --outdir OUT [-- model_to_bricks options]
The code lives in image_to_bricks/images_to_bricks.py."""

from image_to_bricks.images_to_bricks import main

if __name__ == "__main__":
    main()
