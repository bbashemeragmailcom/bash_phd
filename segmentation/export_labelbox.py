import os
import json
import logging
import urllib.request
import requests

from pathlib import Path

import labelbox as lb
from labelbox import StreamType

from dotenv import load_dotenv

from PIL import Image
from tqdm import tqdm


# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()


API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")


if not API_KEY:
    raise SystemExit(
        "API_KEY is not set."
    )


if not PROJECT_ID:
    raise SystemExit(
        "PROJECT_ID is not set."
    )


# ==========================================================
# Labelbox Client
# ==========================================================

client = lb.Client(API_KEY)

project = client.get_project(
    project_id=PROJECT_ID
)


# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = "labelbox_export"

IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

OBJECT_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "object_masks"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)


# ==========================================================
# Create Directories
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    OBJECT_MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)


# ==========================================================
# Export Parameters
# ==========================================================

EXPORT_PARAMS = {

    "attachments": True,
    "data_row_details": True,
    "project_details": True,
    "label_details": True

}


# ==========================================================
# Logger
# ==========================================================

logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(message)s"
    )

)


# ==========================================================
# Utility Functions
# ==========================================================

def save_json(
        data,
        output_path
):

    with open(
            output_path,
            "w"
    ) as f:

        json.dump(

            data,
            f,
            indent=4

        )


def download_image(
        image_url,
        image_path
):
    """
    Download the CT slice.
    """

    response = requests.get(
        image_url
    )

    response.raise_for_status()


    with open(
            image_path,
            "wb"
    ) as f:

        f.write(
            response.content
        )


def download_mask(
        mask_url,
        output_path
):
    """
    Download an individual object mask
    using the Labelbox API headers.
    """

    request = urllib.request.Request(

        mask_url,

        headers=client.headers

    )


    with urllib.request.urlopen(
            request
    ) as response, open(
            output_path,
            "wb"
    ) as f:

        f.write(
            response.read()
        )


# ==========================================================
# Metadata Utilities
# ==========================================================

def initialize_metadata():
    """
    Creates the metadata structure.
    """

    metadata = {

        "background": {

            "id": 0,
            "name": "background"

        },

        "classes": {}

    }


    return metadata


def add_class_to_metadata(
        metadata,
        class_name
):
    """
    Adds a class to metadata.json
    only if it does not already exist.
    """

    classes = metadata["classes"]


    if class_name not in classes:

        class_id = (

            len(classes) + 1

        )


        classes[class_name] = {

            "id": class_id,
            "name": class_name

        }


    return metadata


def build_class_name_to_id(
        metadata
):
    """
    Creates:
    {
        class_name: class_id
    }
    """

    mapping = {

        "background": 0

    }


    for class_name, info in (

            metadata["classes"].items()

    ):

        mapping[class_name] = (

            info["id"]

        )


    return mapping

# ==========================================================
# Filename Utilities
# ==========================================================

def sanitize_filename(value):
    """
    Convert a class name or identifier into a safe filename.
    """

    value = str(value).strip().lower()

    safe_characters = []

    for character in value:

        if character.isalnum():
            safe_characters.append(character)

        elif character in {" ", "-", "_"}:
            safe_characters.append("_")

    sanitized = "".join(safe_characters)

    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    sanitized = sanitized.strip("_")

    return sanitized or "unnamed"


def image_stem_from_external_id(external_id):
    """
    Return a safe image stem without its extension.
    """

    return sanitize_filename(
        Path(external_id).stem
    )


# ==========================================================
# Export Record Utilities
# ==========================================================

def get_project_labels(record):
    """
    Return all labels belonging to PROJECT_ID.

    Falls back to all project entries when PROJECT_ID
    is not found in the record.
    """

    projects = record.get(
        "projects",
        {}
    )

    project_data = projects.get(
        PROJECT_ID
    )

    if project_data is not None:

        return project_data.get(
            "labels",
            []
        )

    labels = []

    for data in projects.values():

        labels.extend(
            data.get(
                "labels",
                []
            )
        )

    return labels


def get_segmentation_objects(record):
    """
    Return every ImageSegmentationMask object
    from all labels associated with a data row.
    """

    segmentation_objects = []

    for label in get_project_labels(record):

        annotations = label.get(
            "annotations",
            {}
        )

        objects = annotations.get(
            "objects",
            []
        )

        for obj in objects:

            if (
                obj.get("annotation_kind")
                == "ImageSegmentationMask"
            ):
                segmentation_objects.append(obj)

    return segmentation_objects


# ==========================================================
# Labelbox Export
# ==========================================================

# def run_labelbox_export():
#     """
#     Run the Labelbox project export and collect
#     every result record into memory.
#     """

#     logging.info(
#         "Starting Labelbox project export."
#     )

#     export_task = project.export(
#         params=EXPORT_PARAMS
#     )

#     export_task.wait_till_done()

#     if export_task.errors:

#         raise RuntimeError(
#             "Labelbox export failed with errors: "
#             f"{export_task.errors}"
#         )

#     stream = export_task.get_buffered_stream(
#         stream_type=StreamType.RESULT
#     )

#     records = []

#     for data_row in tqdm(
#         stream,
#         desc="Reading Labelbox export"
#     ):

#         records.append(
#             data_row.json
#         )

#     logging.info(
#         "Received %d export records.",
#         len(records)
#     )

#     return records

# def run_labelbox_export():
#     """
#     Run the Labelbox export and collect all records.
#     """

#     logging.info("Starting Labelbox export...")

#     export_task = project.export(
#         params=EXPORT_PARAMS
#     )

#     export_task.wait_till_done()

#     records = []

#     result_stream = export_task.get_buffered_stream(
#         stream_type=StreamType.RESULT
#     )

#     for result in tqdm(
#         result_stream,
#         desc="Reading export results"
#     ):

#         records.append(result.json)

#     # Print any export errors
#     error_stream = export_task.get_buffered_stream(
#         stream_type=StreamType.ERRORS
#     )

#     errors = list(error_stream)

#     if errors:

#         print("\nExport Errors:")

#         for error in errors:
#             print(error.json)

#     logging.info(
#         "Received %d records.",
#         len(records)
#     )

#     return records

def run_labelbox_export():
    """
    Runs Labelbox export task
    and returns records.
    """


    export_params = {

        "attachments": True,

        "data_row_details": True,

        "project_details": True,

        "label_details": True

    }


    logging.info(
        "Starting Labelbox export..."
    )


    export_task = project.export(
        params=export_params
    )


    export_task.wait_till_done()


    stream = (
        export_task
        .get_buffered_stream(
            stream_type=lb.StreamType.RESULT
        )
    )


    records = []


    for item in tqdm(
        stream,
        desc="Reading Labelbox export"
    ):


        records.append(
            item.json
        )


    logging.info(
        f"Exported {len(records)} records"
    )


    return records


# ==========================================================
# Build Complete Metadata
# ==========================================================

def build_metadata_from_records(records):
    """
    Build the complete class metadata before processing masks.

    Assigning IDs in a separate pass ensures that every
    annotation_info.json uses a consistent global class ID.
    """

    metadata = initialize_metadata()

    class_names = set()

    for record in records:

        for obj in get_segmentation_objects(
                record
        ):

            class_name = (
                obj.get("value")
                or obj.get("name")
            )

            if class_name:
                class_names.add(
                    class_name
                )

    # Sorting makes class IDs deterministic across exports.
    for class_name in sorted(
            class_names
    ):

        add_class_to_metadata(
            metadata,
            class_name
        )

    logging.info(
        "Discovered %d foreground classes.",
        len(metadata["classes"])
    )

    return metadata


# ==========================================================
# Image Download Processing
# ==========================================================

def process_image_download(record):
    """
    Download the original image for one Labelbox record.

    Returns:
        image_path when successful,
        otherwise None.
    """

    data_row = record.get(
        "data_row",
        {}
    )

    external_id = data_row.get(
        "external_id"
    )

    image_url = data_row.get(
        "row_data"
    )

    if not external_id:

        logging.warning(
            "Skipping data row without external_id."
        )

        return None

    if not image_url:

        logging.warning(
            "No image URL found for %s.",
            external_id
        )

        return None

    image_suffix = (
        Path(external_id).suffix
        or ".png"
    )

    image_filename = (
        image_stem_from_external_id(
            external_id
        )
        + image_suffix.lower()
    )

    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )

    if (
        os.path.exists(image_path)
        and os.path.getsize(image_path) > 0
    ):

        logging.info(
            "Image already exists: %s",
            image_path
        )

        return image_path

    try:

        download_image(
            image_url,
            image_path
        )

        logging.info(
            "Downloaded image: %s",
            image_path
        )

        return image_path

    except Exception as error:

        logging.exception(
            "Failed to download image %s: %s",
            external_id,
            error
        )

        return None


# ==========================================================
# Individual Object Mask Processing
# ==========================================================

def process_object_masks(
        record,
        class_name_to_id
):
    """
    Download all individual object masks for one image.

    Directory example:

    object_masks/
        15467/
            intracerebral_hemorrhage_001.png
            intracerebral_hemorrhage_002.png
            mass_effect_001.png
            annotation_info.json

    Returns:
        Dictionary describing the downloaded masks.
    """

    data_row = record.get(
        "data_row",
        {}
    )

    external_id = data_row.get(
        "external_id"
    )

    if not external_id:

        return None

    image_stem = image_stem_from_external_id(
        external_id
    )

    image_mask_dir = os.path.join(
        OBJECT_MASK_DIR,
        image_stem
    )

    os.makedirs(
        image_mask_dir,
        exist_ok=True
    )

    segmentation_objects = (
        get_segmentation_objects(
            record
        )
    )

    annotation_info = {
        "external_id": external_id,
        "image_stem": image_stem,
        "objects": []
    }

    class_instance_counts = {}

    for obj in segmentation_objects:

        class_name = (
            obj.get("value")
            or obj.get("name")
        )

        if not class_name:

            logging.warning(
                "Skipping unnamed mask object for %s.",
                external_id
            )

            continue

        class_id = class_name_to_id.get(
            class_name
        )

        if class_id is None:

            logging.warning(
                "No class ID found for %s in %s.",
                class_name,
                external_id
            )

            continue

        mask_url = (
            obj.get(
                "mask",
                {}
            ).get(
                "url"
            )
        )

        if not mask_url:

            logging.warning(
                "No individual mask URL for class %s in %s.",
                class_name,
                external_id
            )

            continue

        safe_class_name = sanitize_filename(
            class_name
        )

        instance_number = (
            class_instance_counts.get(
                safe_class_name,
                0
            )
            + 1
        )

        class_instance_counts[
            safe_class_name
        ] = instance_number

        mask_filename = (
            f"{safe_class_name}_"
            f"{instance_number:03d}.png"
        )

        mask_path = os.path.join(
            image_mask_dir,
            mask_filename
        )

        download_status = "existing"

        if not (
            os.path.exists(mask_path)
            and os.path.getsize(mask_path) > 0
        ):

            try:

                download_mask(
                    mask_url,
                    mask_path
                )

                download_status = "downloaded"

            except Exception as error:

                logging.exception(
                    "Failed to download mask %s for %s: %s",
                    class_name,
                    external_id,
                    error
                )

                annotation_info["objects"].append(
                    {
                        "class_name": class_name,
                        "class_id": class_id,
                        "feature_id": obj.get(
                            "feature_id"
                        ),
                        "feature_schema_id": obj.get(
                            "feature_schema_id"
                        ),
                        "mask_filename": mask_filename,
                        "download_status": "failed",
                        "error": str(error)
                    }
                )

                continue

        annotation_info["objects"].append(
            {
                "class_name": class_name,
                "class_id": class_id,
                "feature_id": obj.get(
                    "feature_id"
                ),
                "feature_schema_id": obj.get(
                    "feature_schema_id"
                ),
                "mask_filename": mask_filename,
                "download_status": download_status
            }
        )

        logging.info(
            "%s object mask: %s",
            download_status.capitalize(),
            mask_path
        )

    annotation_info["num_objects"] = len(
        annotation_info["objects"]
    )

    annotation_info_path = os.path.join(
        image_mask_dir,
        "annotation_info.json"
    )

    save_json(
        annotation_info,
        annotation_info_path
    )

    return annotation_info


# ==========================================================
# Process All Export Records
# ==========================================================

def process_export_records(
        records,
        metadata
):
    """
    Download all images and individual object masks.

    Returns a dataset manifest that Part 3 will save.
    """

    class_name_to_id = (
        build_class_name_to_id(
            metadata
        )
    )

    manifest = []

    for record in tqdm(
        records,
        desc="Downloading images and masks"
    ):

        data_row = record.get(
            "data_row",
            {}
        )

        external_id = data_row.get(
            "external_id"
        )

        logging.info(
            "Processing data row: %s",
            external_id
        )

        image_path = process_image_download(
            record
        )

        annotation_info = (
            process_object_masks(
                record,
                class_name_to_id
            )
        )

        manifest.append(
            {
                "data_row_id": data_row.get(
                    "id"
                ),
                "external_id": external_id,
                "image_filename": (
                    os.path.basename(image_path)
                    if image_path
                    else None
                ),
                "mask_directory": (
                    image_stem_from_external_id(
                        external_id
                    )
                    if external_id
                    else None
                ),
                "num_objects": (
                    annotation_info.get(
                        "num_objects",
                        0
                    )
                    if annotation_info
                    else 0
                ),
                "image_downloaded": (
                    image_path is not None
                )
            }
        )

    return manifest

# ==========================================================
# Additional Imports
# ==========================================================

import argparse
import colorsys

import numpy as np


# ==========================================================
# Output File Paths
# ==========================================================

RAW_EXPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "export.json"
)

METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "manifest.json"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "export_summary.json"
)


# ==========================================================
# Overlay Configuration
# ==========================================================

OVERLAY_ALPHA = 0.45


# ==========================================================
# Class Color Utilities
# ==========================================================

def generate_class_colors(metadata):
    """
    Generate a deterministic RGB visualization color
    for every foreground class.

    These colors are used only for overlays. They are not
    used to generate the training semantic masks.
    """

    class_colors = {
        "background": [0, 0, 0]
    }

    classes = sorted(
        metadata["classes"].items(),
        key=lambda item: item[1]["id"]
    )

    total_classes = max(
        len(classes),
        1
    )

    for index, (class_name, class_info) in enumerate(
            classes
    ):

        hue = index / total_classes

        red, green, blue = colorsys.hsv_to_rgb(
            hue,
            0.80,
            1.00
        )

        class_colors[class_name] = [
            int(red * 255),
            int(green * 255),
            int(blue * 255)
        ]

        class_info["overlay_rgb"] = (
            class_colors[class_name]
        )

    return class_colors


# ==========================================================
# Object Mask Loading
# ==========================================================

def load_binary_object_mask(mask_path):
    """
    Load an individual Labelbox object mask as a
    boolean NumPy array.

    Labelbox object masks may be stored as:
    - grayscale images,
    - RGB images,
    - RGBA images.

    Any non-zero pixel is treated as part of the object.
    """

    with Image.open(mask_path) as mask_image:

        mask_array = np.array(
            mask_image
        )

    if mask_array.ndim == 2:

        binary_mask = mask_array > 0

    elif mask_array.ndim == 3:

        if mask_array.shape[2] == 4:

            rgb_channels = mask_array[:, :, :3]
            alpha_channel = mask_array[:, :, 3]

            binary_mask = (
                np.any(
                    rgb_channels > 0,
                    axis=-1
                )
                | (
                    alpha_channel > 0
                )
            )

        else:

            binary_mask = np.any(
                mask_array[:, :, :3] > 0,
                axis=-1
            )

    else:

        raise ValueError(
            f"Unsupported mask dimensions "
            f"{mask_array.shape} for {mask_path}"
        )

    return binary_mask


# ==========================================================
# Object Mask Validation
# ==========================================================

def validate_object_mask(
        mask_path,
        expected_size=None
):
    """
    Validate a downloaded object mask.

    Returns:
        {
            "valid": bool,
            "width": int or None,
            "height": int or None,
            "foreground_pixels": int,
            "error": str or None
        }
    """

    result = {
        "valid": False,
        "width": None,
        "height": None,
        "foreground_pixels": 0,
        "error": None
    }

    if not os.path.exists(mask_path):

        result["error"] = (
            "Mask file does not exist."
        )

        return result

    if os.path.getsize(mask_path) == 0:

        result["error"] = (
            "Mask file is empty."
        )

        return result

    try:

        with Image.open(mask_path) as image:

            result["width"] = image.width
            result["height"] = image.height

        if expected_size is not None:

            if (
                result["width"],
                result["height"]
            ) != expected_size:

                result["error"] = (
                    "Mask dimensions do not match "
                    f"the source image. Mask: "
                    f"{result['width']}x"
                    f"{result['height']}; image: "
                    f"{expected_size[0]}x"
                    f"{expected_size[1]}."
                )

                return result

        binary_mask = load_binary_object_mask(
            mask_path
        )

        result["foreground_pixels"] = int(
            binary_mask.sum()
        )

        if result["foreground_pixels"] == 0:

            result["error"] = (
                "Mask contains no foreground pixels."
            )

            return result

        result["valid"] = True

    except Exception as error:

        result["error"] = str(error)

    return result


# ==========================================================
# Semantic Preview Creation
# ==========================================================

def build_semantic_preview(
        image_size,
        annotation_info,
        image_mask_dir
):
    """
    Combine individual object masks into a semantic preview.

    The resulting array contains:
        0 = background
        1...N = class IDs

    This preview is used for validation and overlays.
    The final training semantic mask can also be generated
    using the same logic inside build_dataset.py.
    """

    width, height = image_size

    semantic_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    valid_objects = []
    invalid_objects = []

    for object_info in annotation_info.get(
            "objects",
            []
    ):

        if object_info.get(
                "download_status"
        ) == "failed":

            invalid_objects.append(
                {
                    **object_info,
                    "validation_error": (
                        object_info.get(
                            "error",
                            "Mask download failed."
                        )
                    )
                }
            )

            continue

        mask_filename = object_info.get(
            "mask_filename"
        )

        if not mask_filename:

            invalid_objects.append(
                {
                    **object_info,
                    "validation_error": (
                        "mask_filename is missing."
                    )
                }
            )

            continue

        mask_path = os.path.join(
            image_mask_dir,
            mask_filename
        )

        validation = validate_object_mask(
            mask_path,
            expected_size=image_size
        )

        if not validation["valid"]:

            invalid_objects.append(
                {
                    **object_info,
                    "validation_error": (
                        validation["error"]
                    )
                }
            )

            continue

        binary_mask = load_binary_object_mask(
            mask_path
        )

        class_id = int(
            object_info["class_id"]
        )

        semantic_mask[
            binary_mask
        ] = class_id

        valid_objects.append(
            {
                **object_info,
                "foreground_pixels": (
                    validation[
                        "foreground_pixels"
                    ]
                )
            }
        )

    return (
        semantic_mask,
        valid_objects,
        invalid_objects
    )


# ==========================================================
# Overlay Creation
# ==========================================================

def create_overlay(
        image_path,
        annotation_info,
        metadata,
        output_path
):
    """
    Create a visualization overlay using the downloaded
    individual masks.

    Returns a dictionary containing overlay statistics.
    """

    with Image.open(image_path) as image:

        source_image = image.convert(
            "RGB"
        )

    image_array = np.array(
        source_image
    )

    image_size = source_image.size

    image_stem = annotation_info[
        "image_stem"
    ]

    image_mask_dir = os.path.join(
        OBJECT_MASK_DIR,
        image_stem
    )

    (
        semantic_mask,
        valid_objects,
        invalid_objects
    ) = build_semantic_preview(
        image_size=image_size,
        annotation_info=annotation_info,
        image_mask_dir=image_mask_dir
    )

    color_mask = np.zeros_like(
        image_array,
        dtype=np.uint8
    )

    for class_name, class_info in (
            metadata["classes"].items()
    ):

        class_id = class_info["id"]

        class_color = class_info.get(
            "overlay_rgb",
            [255, 255, 255]
        )

        class_pixels = (
            semantic_mask == class_id
        )

        color_mask[
            class_pixels
        ] = class_color

    foreground = semantic_mask > 0

    overlay_array = image_array.copy()

    if np.any(foreground):

        blended_pixels = (
            (
                1.0 - OVERLAY_ALPHA
            )
            * image_array[
                foreground
            ].astype(
                np.float32
            )
            +
            OVERLAY_ALPHA
            * color_mask[
                foreground
            ].astype(
                np.float32
            )
        )

        overlay_array[
            foreground
        ] = np.clip(
            blended_pixels,
            0,
            255
        ).astype(
            np.uint8
        )

    overlay_image = Image.fromarray(
        overlay_array
    )

    overlay_image.save(
        output_path
    )

    return {
        "valid_objects": valid_objects,
        "invalid_objects": invalid_objects,
        "semantic_classes": [
            int(value)
            for value in np.unique(
                semantic_mask
            )
        ],
        "foreground_pixels": int(
            foreground.sum()
        )
    }


# ==========================================================
# Overlay Processing
# ==========================================================

def generate_all_overlays(
        manifest,
        metadata
):
    """
    Generate an overlay for every successfully
    downloaded source image.
    """

    overlay_results = []

    for item in tqdm(
        manifest,
        desc="Generating overlays"
    ):

        external_id = item.get(
            "external_id"
        )

        image_filename = item.get(
            "image_filename"
        )

        mask_directory = item.get(
            "mask_directory"
        )

        result = {
            "external_id": external_id,
            "overlay_created": False,
            "foreground_pixels": 0,
            "valid_objects": 0,
            "invalid_objects": 0,
            "error": None
        }

        if not image_filename:

            result["error"] = (
                "Source image was not downloaded."
            )

            overlay_results.append(
                result
            )

            continue

        if not mask_directory:

            result["error"] = (
                "Object mask directory is missing."
            )

            overlay_results.append(
                result
            )

            continue

        image_path = os.path.join(
            IMAGE_DIR,
            image_filename
        )

        annotation_info_path = os.path.join(
            OBJECT_MASK_DIR,
            mask_directory,
            "annotation_info.json"
        )

        if not os.path.exists(
                annotation_info_path
        ):

            result["error"] = (
                "annotation_info.json was not found."
            )

            overlay_results.append(
                result
            )

            continue

        try:

            with open(
                    annotation_info_path,
                    "r"
            ) as file:

                annotation_info = json.load(
                    file
                )

            overlay_filename = (
                f"{mask_directory}_overlay.png"
            )

            overlay_path = os.path.join(
                OVERLAY_DIR,
                overlay_filename
            )

            overlay_statistics = create_overlay(
                image_path=image_path,
                annotation_info=annotation_info,
                metadata=metadata,
                output_path=overlay_path
            )

            result.update(
                {
                    "overlay_created": True,
                    "overlay_filename": (
                        overlay_filename
                    ),
                    "foreground_pixels": (
                        overlay_statistics[
                            "foreground_pixels"
                        ]
                    ),
                    "valid_objects": len(
                        overlay_statistics[
                            "valid_objects"
                        ]
                    ),
                    "invalid_objects": len(
                        overlay_statistics[
                            "invalid_objects"
                        ]
                    ),
                    "semantic_classes": (
                        overlay_statistics[
                            "semantic_classes"
                        ]
                    )
                }
            )

            if overlay_statistics[
                    "invalid_objects"
            ]:

                result[
                    "invalid_object_details"
                ] = overlay_statistics[
                    "invalid_objects"
                ]

        except Exception as error:

            logging.exception(
                "Failed to create overlay for %s: %s",
                external_id,
                error
            )

            result["error"] = str(error)

        overlay_results.append(
            result
        )

    return overlay_results


# ==========================================================
# Export Summary
# ==========================================================

def build_export_summary(
        records,
        metadata,
        manifest,
        overlay_results
):
    """
    Build a concise summary of the completed export.
    """

    downloaded_images = sum(
        1
        for item in manifest
        if item.get("image_downloaded")
    )

    total_objects = sum(
        item.get("num_objects", 0)
        for item in manifest
    )

    successful_overlays = sum(
        1
        for item in overlay_results
        if item.get("overlay_created")
    )

    valid_object_masks = sum(
        item.get("valid_objects", 0)
        for item in overlay_results
    )

    invalid_object_masks = sum(
        item.get("invalid_objects", 0)
        for item in overlay_results
    )

    images_with_foreground = sum(
        1
        for item in overlay_results
        if item.get(
            "foreground_pixels",
            0
        ) > 0
    )

    return {
        "project_id": PROJECT_ID,
        "output_directory": OUTPUT_DIR,
        "total_export_records": len(
            records
        ),
        "downloaded_images": (
            downloaded_images
        ),
        "foreground_classes": len(
            metadata["classes"]
        ),
        "total_annotation_objects": (
            total_objects
        ),
        "valid_object_masks": (
            valid_object_masks
        ),
        "invalid_object_masks": (
            invalid_object_masks
        ),
        "overlays_created": (
            successful_overlays
        ),
        "images_with_foreground_masks": (
            images_with_foreground
        ),
        "files": {
            "raw_export": (
                os.path.basename(
                    RAW_EXPORT_PATH
                )
            ),
            "metadata": (
                os.path.basename(
                    METADATA_PATH
                )
            ),
            "manifest": (
                os.path.basename(
                    MANIFEST_PATH
                )
            ),
            "summary": (
                os.path.basename(
                    SUMMARY_PATH
                )
            )
        }
    }


# ==========================================================
# Console Summary
# ==========================================================

def print_export_summary(summary):
    """
    Print the final export statistics.
    """

    print("\n")
    print("=" * 60)
    print("LABELBOX EXPORT COMPLETED")
    print("=" * 60)

    print(
        "Export records:",
        summary["total_export_records"]
    )

    print(
        "Images downloaded:",
        summary["downloaded_images"]
    )

    print(
        "Foreground classes:",
        summary["foreground_classes"]
    )

    print(
        "Annotation objects:",
        summary["total_annotation_objects"]
    )

    print(
        "Valid object masks:",
        summary["valid_object_masks"]
    )

    print(
        "Invalid object masks:",
        summary["invalid_object_masks"]
    )

    print(
        "Overlays created:",
        summary["overlays_created"]
    )

    print(
        "Images with foreground:",
        summary[
            "images_with_foreground_masks"
        ]
    )

    print(
        "Output directory:",
        summary["output_directory"]
    )

    print("=" * 60)


# ==========================================================
# Command-Line Arguments
# ==========================================================

def parse_arguments():
    """
    Parse optional command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Export Labelbox segmentation data and "
            "download individual object masks."
        )
    )

    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help=(
            "Download images and object masks without "
            "creating visualization overlays."
        )
    )

    return parser.parse_args()


# ==========================================================
# Main Pipeline
# ==========================================================

def main():
    """
    Run the complete Labelbox export pipeline.
    """

    args = parse_arguments()

    logging.info(
        "Export output directory: %s",
        OUTPUT_DIR
    )

    # ------------------------------------------------------
    # 1. Run Labelbox export
    # ------------------------------------------------------

    records = run_labelbox_export()

    if not records:

        raise RuntimeError(
            "The Labelbox export returned no records."
        )

    # ------------------------------------------------------
    # 2. Save raw export
    # ------------------------------------------------------

    save_json(
        records,
        RAW_EXPORT_PATH
    )

    logging.info(
        "Saved raw export: %s",
        RAW_EXPORT_PATH
    )

    # ------------------------------------------------------
    # 3. Build and save class metadata
    # ------------------------------------------------------

    metadata = build_metadata_from_records(
        records
    )

    generate_class_colors(
        metadata
    )

    metadata["project_id"] = PROJECT_ID

    metadata["num_foreground_classes"] = len(
        metadata["classes"]
    )

    metadata["mask_format"] = (
        "individual_binary_object_masks"
    )

    save_json(
        metadata,
        METADATA_PATH
    )

    logging.info(
        "Saved metadata: %s",
        METADATA_PATH
    )

    # ------------------------------------------------------
    # 4. Download images and object masks
    # ------------------------------------------------------

    manifest = process_export_records(
        records=records,
        metadata=metadata
    )

    save_json(
        manifest,
        MANIFEST_PATH
    )

    logging.info(
        "Saved manifest: %s",
        MANIFEST_PATH
    )

    # ------------------------------------------------------
    # 5. Generate overlays
    # ------------------------------------------------------

    if args.skip_overlays:

        logging.info(
            "Overlay generation was skipped."
        )

        overlay_results = []

    else:

        overlay_results = generate_all_overlays(
            manifest=manifest,
            metadata=metadata
        )

    # ------------------------------------------------------
    # 6. Build and save summary
    # ------------------------------------------------------

    summary = build_export_summary(
        records=records,
        metadata=metadata,
        manifest=manifest,
        overlay_results=overlay_results
    )

    summary["overlay_results"] = (
        overlay_results
    )

    save_json(
        summary,
        SUMMARY_PATH
    )

    logging.info(
        "Saved export summary: %s",
        SUMMARY_PATH
    )

    print_export_summary(
        summary
    )


# ==========================================================
# Script Entry Point
# ==========================================================

if __name__ == "__main__":
    main()