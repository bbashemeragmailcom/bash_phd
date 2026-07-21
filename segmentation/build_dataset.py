import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ==========================================================
# Configuration
# ==========================================================

EXPORT_DIR = Path("labelbox_export")
OUTPUT_DIR = Path("stroke_dataset")

SOURCE_IMAGE_DIR = EXPORT_DIR / "images"
SOURCE_OBJECT_MASK_DIR = EXPORT_DIR / "object_masks"

SOURCE_METADATA_PATH = EXPORT_DIR / "metadata.json"
SOURCE_MANIFEST_PATH = EXPORT_DIR / "manifest.json"

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_MASK_DIR = OUTPUT_DIR / "masks"
OUTPUT_OVERLAY_DIR = OUTPUT_DIR / "overlays"
OUTPUT_NPZ_DIR = OUTPUT_DIR / "npz"
OUTPUT_SPLIT_DIR = OUTPUT_DIR / "splits"

OUTPUT_METADATA_PATH = OUTPUT_DIR / "metadata.json"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"
OUTPUT_STATISTICS_PATH = OUTPUT_DIR / "class_statistics.json"
OUTPUT_REPORT_PATH = OUTPUT_DIR / "preprocessing_report.json"
OUTPUT_YAML_PATH = OUTPUT_DIR / "dataset.yaml"


# ==========================================================
# Split Configuration
# ==========================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ==========================================================
# Processing Configuration
# ==========================================================

CREATE_OVERLAYS = True
CREATE_NPZ = True
COPY_IMAGES = True

OVERWRITE_EXISTING = True

# Resize object masks to the source image size when their
# dimensions differ.
RESIZE_MISMATCHED_MASKS = False

# Supported options:
# "last_wins"
# "first_wins"
# "highest_class_id"
OVERLAP_POLICY = "last_wins"

OVERLAY_ALPHA = 0.45


# ==========================================================
# Logging
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
# JSON Utilities
# ==========================================================

def load_json(path):
    """
    Load and return a JSON file.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    """
    Save data as formatted JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4
        )


# ==========================================================
# Directory Setup
# ==========================================================

def create_output_directories():
    """
    Create all dataset output directories.
    """

    directories = [
        OUTPUT_DIR,
        OUTPUT_IMAGE_DIR,
        OUTPUT_MASK_DIR,
        OUTPUT_OVERLAY_DIR,
        OUTPUT_NPZ_DIR,
        OUTPUT_SPLIT_DIR
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ==========================================================
# Configuration Validation
# ==========================================================

def validate_configuration():
    """
    Validate paths, split ratios and processing settings.
    """

    required_paths = [
        SOURCE_IMAGE_DIR,
        SOURCE_OBJECT_MASK_DIR,
        SOURCE_METADATA_PATH
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required input path does not exist: {path}"
            )

    ratio_sum = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if not np.isclose(
        ratio_sum,
        1.0
    ):

        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO "
            f"must equal 1.0. Current total: {ratio_sum}"
        )

    valid_overlap_policies = {
        "last_wins",
        "first_wins",
        "highest_class_id"
    }

    if OVERLAP_POLICY not in valid_overlap_policies:

        raise ValueError(
            f"Invalid overlap policy: {OVERLAP_POLICY}. "
            f"Expected one of {sorted(valid_overlap_policies)}."
        )


# ==========================================================
# Metadata Utilities
# ==========================================================

def load_metadata():
    """
    Load Labelbox export metadata.
    """

    metadata = load_json(
        SOURCE_METADATA_PATH
    )

    if "classes" not in metadata:

        raise ValueError(
            "metadata.json does not contain a 'classes' section."
        )

    return metadata


def get_class_records(metadata):
    """
    Return class records sorted by class ID.
    """

    class_records = []

    for class_name, information in metadata[
            "classes"
    ].items():

        class_records.append(
            {
                "name": class_name,
                "id": int(
                    information["id"]
                ),
                "overlay_rgb": information.get(
                    "overlay_rgb"
                )
            }
        )

    class_records.sort(
        key=lambda item: item["id"]
    )

    return class_records


def build_class_lookup(metadata):
    """
    Return class ID and class-name lookup dictionaries.
    """

    id_to_name = {
        0: "background"
    }

    name_to_id = {
        "background": 0
    }

    for class_name, information in metadata[
            "classes"
    ].items():

        class_id = int(
            information["id"]
        )

        id_to_name[class_id] = class_name
        name_to_id[class_name] = class_id

    return id_to_name, name_to_id


# ==========================================================
# Manifest Discovery
# ==========================================================

def discover_records():
    """
    Load manifest.json when available.

    If the manifest does not exist, discover records from
    the object_masks directory.
    """

    if SOURCE_MANIFEST_PATH.exists():

        manifest = load_json(
            SOURCE_MANIFEST_PATH
        )

        logging.info(
            "Loaded %d records from manifest.json.",
            len(manifest)
        )

        return manifest

    logging.warning(
        "manifest.json was not found. "
        "Discovering records from object_masks."
    )

    records = []

    for mask_directory in sorted(
            SOURCE_OBJECT_MASK_DIR.iterdir()
    ):

        if not mask_directory.is_dir():
            continue

        annotation_path = (
            mask_directory
            / "annotation_info.json"
        )

        if not annotation_path.exists():
            continue

        annotation_info = load_json(
            annotation_path
        )

        external_id = annotation_info.get(
            "external_id"
        )

        image_filename = find_source_image_filename(
            external_id=external_id,
            image_stem=mask_directory.name
        )

        records.append(
            {
                "external_id": external_id,
                "image_filename": image_filename,
                "mask_directory": mask_directory.name,
                "num_objects": len(
                    annotation_info.get(
                        "objects",
                        []
                    )
                ),
                "image_downloaded": (
                    image_filename is not None
                )
            }
        )

    logging.info(
        "Discovered %d records.",
        len(records)
    )

    return records


# ==========================================================
# Image Discovery and Loading
# ==========================================================

def find_source_image_filename(
        external_id,
        image_stem
):
    """
    Find the corresponding source image filename.
    """

    if external_id:

        exact_path = (
            SOURCE_IMAGE_DIR
            / Path(external_id).name
        )

        if exact_path.exists():
            return exact_path.name

    supported_extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp"
    ]

    for extension in supported_extensions:

        candidate = (
            SOURCE_IMAGE_DIR
            / f"{image_stem}{extension}"
        )

        if candidate.exists():
            return candidate.name

    matches = list(
        SOURCE_IMAGE_DIR.glob(
            f"{image_stem}.*"
        )
    )

    if matches:
        return matches[0].name

    return None


def load_source_image(image_path):
    """
    Load a source image as an RGB NumPy array.
    """

    with Image.open(image_path) as image:

        rgb_image = image.convert(
            "RGB"
        )

        image_array = np.array(
            rgb_image,
            dtype=np.uint8
        )

    return image_array


# ==========================================================
# Individual Object Mask Loading
# ==========================================================

def load_binary_object_mask(
        mask_path,
        expected_size=None
):
    """
    Load an individual Labelbox object mask.

    Returns a boolean array where True represents the object.

    Supports:
    - grayscale masks,
    - RGB masks,
    - RGBA masks,
    - palette masks.
    """

    with Image.open(mask_path) as image:

        original_mode = image.mode

        mask_array = np.array(
            image
        )

    if mask_array.ndim == 2:

        binary_mask = (
            mask_array > 0
        )

    elif (
        mask_array.ndim == 3
        and mask_array.shape[2] == 4
    ):

        rgb = mask_array[:, :, :3]
        alpha = mask_array[:, :, 3]

        alpha_has_transparency = (
            np.any(alpha == 0)
            and np.any(alpha > 0)
        )

        if alpha_has_transparency:

            binary_mask = (
                alpha > 0
            )

        else:

            binary_mask = np.any(
                rgb > 0,
                axis=-1
            )

    elif (
        mask_array.ndim == 3
        and mask_array.shape[2] >= 3
    ):

        binary_mask = np.any(
            mask_array[:, :, :3] > 0,
            axis=-1
        )

    else:

        raise ValueError(
            f"Unsupported object-mask format at {mask_path}. "
            f"Mode: {original_mode}; shape: {mask_array.shape}"
        )

    if expected_size is not None:

        expected_width, expected_height = (
            expected_size
        )

        actual_height, actual_width = (
            binary_mask.shape
        )

        dimensions_match = (
            actual_width == expected_width
            and actual_height == expected_height
        )

        if not dimensions_match:

            if not RESIZE_MISMATCHED_MASKS:

                raise ValueError(
                    f"Mask dimensions do not match image dimensions. "
                    f"Mask: {actual_width}x{actual_height}; "
                    f"image: {expected_width}x{expected_height}; "
                    f"path: {mask_path}"
                )

            binary_image = Image.fromarray(
                binary_mask.astype(
                    np.uint8
                )
                * 255
            )

            binary_image = binary_image.resize(
                (
                    expected_width,
                    expected_height
                ),
                resample=Image.Resampling.NEAREST
            )

            binary_mask = (
                np.array(binary_image) > 0
            )

    return binary_mask


# ==========================================================
# Semantic Mask Construction
# ==========================================================

def apply_object_to_semantic_mask(
        semantic_mask,
        binary_mask,
        class_id
):
    """
    Apply an object mask using the selected overlap policy.
    """

    if OVERLAP_POLICY == "last_wins":

        semantic_mask[
            binary_mask
        ] = class_id

    elif OVERLAP_POLICY == "first_wins":

        writable_pixels = (
            binary_mask
            & (
                semantic_mask == 0
            )
        )

        semantic_mask[
            writable_pixels
        ] = class_id

    elif OVERLAP_POLICY == "highest_class_id":

        existing_values = semantic_mask[
            binary_mask
        ]

        semantic_mask[
            binary_mask
        ] = np.maximum(
            existing_values,
            class_id
        )


def build_semantic_mask(
        image_shape,
        annotation_info,
        object_mask_directory
):
    """
    Combine individual Labelbox object masks into one
    integer semantic mask.

    Returns:
        semantic_mask
        valid_objects
        invalid_objects
    """

    height, width = image_shape[:2]

    semantic_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    valid_objects = []
    invalid_objects = []

    for object_information in annotation_info.get(
            "objects",
            []
    ):

        class_name = object_information.get(
            "class_name"
        )

        class_id = object_information.get(
            "class_id"
        )

        mask_filename = object_information.get(
            "mask_filename"
        )

        download_status = object_information.get(
            "download_status"
        )

        if download_status == "failed":

            invalid_objects.append(
                {
                    **object_information,
                    "reason": "Mask download failed."
                }
            )

            continue

        if class_id is None:

            invalid_objects.append(
                {
                    **object_information,
                    "reason": "class_id is missing."
                }
            )

            continue

        if not mask_filename:

            invalid_objects.append(
                {
                    **object_information,
                    "reason": "mask_filename is missing."
                }
            )

            continue

        object_mask_path = (
            object_mask_directory
            / mask_filename
        )

        if not object_mask_path.exists():

            invalid_objects.append(
                {
                    **object_information,
                    "reason": (
                        f"Mask file does not exist: "
                        f"{object_mask_path}"
                    )
                }
            )

            continue

        try:

            binary_mask = load_binary_object_mask(
                mask_path=object_mask_path,
                expected_size=(
                    width,
                    height
                )
            )

            foreground_pixels = int(
                binary_mask.sum()
            )

            if foreground_pixels == 0:

                invalid_objects.append(
                    {
                        **object_information,
                        "reason": (
                            "Object mask has no foreground pixels."
                        )
                    }
                )

                continue

            class_id = int(
                class_id
            )

            apply_object_to_semantic_mask(
                semantic_mask=semantic_mask,
                binary_mask=binary_mask,
                class_id=class_id
            )

            valid_objects.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "mask_filename": mask_filename,
                    "foreground_pixels": foreground_pixels
                }
            )

        except Exception as error:

            invalid_objects.append(
                {
                    **object_information,
                    "reason": str(error)
                }
            )

    return (
        semantic_mask,
        valid_objects,
        invalid_objects
    )


# ==========================================================
# Output Saving
# ==========================================================

def save_semantic_mask(
        semantic_mask,
        output_path
):
    """
    Save a semantic mask as a single-channel PNG.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_image = Image.fromarray(
        semantic_mask.astype(
            np.uint8
        ),
        mode="L"
    )

    mask_image.save(
        output_path
    )


def copy_source_image(
        source_path,
        destination_path
):
    """
    Copy a source image to the dataset directory.
    """

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        destination_path.exists()
        and not OVERWRITE_EXISTING
    ):

        return

    shutil.copy2(
        source_path,
        destination_path
    )


def save_npz_file(
        image_array,
        semantic_mask,
        output_path,
        sample_id,
        external_id
):
    """
    Save the image and semantic mask in a compressed NPZ file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        image=image_array.astype(
            np.uint8
        ),
        mask=semantic_mask.astype(
            np.uint8
        ),
        sample_id=np.array(
            sample_id
        ),
        external_id=np.array(
            external_id or ""
        )
    )


# ==========================================================
# Overlay Generation
# ==========================================================

def generate_fallback_color(class_id):
    """
    Generate a deterministic visualization color.
    """

    return [
        int(
            (
                class_id * 67
            )
            % 256
        ),
        int(
            (
                class_id * 137
            )
            % 256
        ),
        int(
            (
                class_id * 197
            )
            % 256
        )
    ]


def build_class_color_lookup(metadata):
    """
    Return class ID to RGB visualization-color mapping.
    """

    class_colors = {
        0: [0, 0, 0]
    }

    for _, information in metadata[
            "classes"
    ].items():

        class_id = int(
            information["id"]
        )

        color = information.get(
            "overlay_rgb"
        )

        if color is None:

            color = generate_fallback_color(
                class_id
            )

        class_colors[class_id] = [
            int(channel)
            for channel in color
        ]

    return class_colors


def create_overlay(
        image_array,
        semantic_mask,
        class_colors,
        output_path
):
    """
    Create and save an image/semantic-mask overlay.
    """

    overlay = image_array.copy()

    foreground = (
        semantic_mask > 0
    )

    color_mask = np.zeros_like(
        image_array,
        dtype=np.uint8
    )

    for class_id, color in class_colors.items():

        if class_id == 0:
            continue

        class_pixels = (
            semantic_mask == class_id
        )

        color_mask[
            class_pixels
        ] = color

    if np.any(foreground):

        blended = (
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

        overlay[
            foreground
        ] = np.clip(
            blended,
            0,
            255
        ).astype(
            np.uint8
        )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )


# ==========================================================
# Class Statistics
# ==========================================================

def initialize_class_statistics(
        id_to_name
):
    """
    Initialize per-class statistics.
    """

    statistics = {}

    for class_id, class_name in sorted(
            id_to_name.items()
    ):

        statistics[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_name,
            "pixel_count": 0,
            "image_count": 0,
            "object_count": 0
        }

    return statistics


def update_class_statistics(
        statistics,
        semantic_mask,
        valid_objects
):
    """
    Update class statistics for one sample.
    """

    unique_ids, counts = np.unique(
        semantic_mask,
        return_counts=True
    )

    for class_id, pixel_count in zip(
            unique_ids,
            counts
    ):

        key = str(
            int(class_id)
        )

        if key not in statistics:

            statistics[key] = {
                "class_id": int(class_id),
                "class_name": f"class_{class_id}",
                "pixel_count": 0,
                "image_count": 0,
                "object_count": 0
            }

        statistics[key]["pixel_count"] += int(
            pixel_count
        )

        statistics[key]["image_count"] += 1

    for object_information in valid_objects:

        key = str(
            int(
                object_information[
                    "class_id"
                ]
            )
        )

        if key in statistics:

            statistics[key]["object_count"] += 1


def finalize_class_statistics(
        statistics
):
    """
    Add pixel percentages to class statistics.
    """

    total_pixels = sum(
        information["pixel_count"]
        for information in statistics.values()
    )

    for information in statistics.values():

        if total_pixels > 0:

            information["pixel_percentage"] = (
                information["pixel_count"]
                / total_pixels
                * 100.0
            )

        else:

            information["pixel_percentage"] = 0.0

    return statistics


# ==========================================================
# Dataset Splitting
# ==========================================================

def split_sample_ids(sample_ids):
    """
    Randomly divide sample IDs into train, validation and test.
    """

    shuffled_ids = list(
        sample_ids
    )

    random_generator = random.Random(
        RANDOM_SEED
    )

    random_generator.shuffle(
        shuffled_ids
    )

    total_samples = len(
        shuffled_ids
    )

    train_count = int(
        total_samples
        * TRAIN_RATIO
    )

    validation_count = int(
        total_samples
        * VAL_RATIO
    )

    test_count = (
        total_samples
        - train_count
        - validation_count
    )

    train_ids = shuffled_ids[
        :train_count
    ]

    validation_ids = shuffled_ids[
        train_count:
        train_count + validation_count
    ]

    test_ids = shuffled_ids[
        train_count + validation_count:
    ]

    assert len(test_ids) == test_count

    return {
        "train": train_ids,
        "val": validation_ids,
        "test": test_ids
    }


def save_split_files(splits):
    """
    Save one text file for each dataset split.
    """

    for split_name, sample_ids in splits.items():

        split_path = (
            OUTPUT_SPLIT_DIR
            / f"{split_name}.txt"
        )

        with open(
                split_path,
                "w",
                encoding="utf-8"
        ) as file:

            for sample_id in sample_ids:
                file.write(
                    f"{sample_id}\n"
                )


# ==========================================================
# Dataset YAML
# ==========================================================

def save_dataset_yaml(
        metadata,
        splits
):
    """
    Save a simple framework-independent dataset YAML file.

    This implementation avoids requiring PyYAML.
    """

    class_records = get_class_records(
        metadata
    )

    lines = [
        f"path: {OUTPUT_DIR.resolve()}",
        "images: images",
        "masks: masks",
        "npz: npz",
        "splits: splits",
        "",
        f"num_classes: {len(class_records) + 1}",
        "",
        "classes:",
        "  0: background"
    ]

    for class_record in class_records:

        lines.append(
            f"  {class_record['id']}: "
            f"{class_record['name']}"
        )

    lines.extend(
        [
            "",
            "split_sizes:",
            f"  train: {len(splits['train'])}",
            f"  val: {len(splits['val'])}",
            f"  test: {len(splits['test'])}",
            "",
            f"random_seed: {RANDOM_SEED}",
            f"overlap_policy: {OVERLAP_POLICY}"
        ]
    )

    with open(
            OUTPUT_YAML_PATH,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(lines)
            + "\n"
        )


# ==========================================================
# Individual Record Processing
# ==========================================================

def process_record(
        record,
        metadata,
        class_colors
):
    """
    Process one image and its individual object masks.
    """

    external_id = record.get(
        "external_id"
    )

    mask_directory_name = record.get(
        "mask_directory"
    )

    if not mask_directory_name:

        if external_id:

            mask_directory_name = (
                Path(external_id).stem
            )

        else:

            raise ValueError(
                "Record contains neither mask_directory "
                "nor external_id."
            )

    sample_id = mask_directory_name

    image_filename = record.get(
        "image_filename"
    )

    if not image_filename:

        image_filename = find_source_image_filename(
            external_id=external_id,
            image_stem=sample_id
        )

    if not image_filename:

        raise FileNotFoundError(
            f"No source image found for sample {sample_id}."
        )

    source_image_path = (
        SOURCE_IMAGE_DIR
        / image_filename
    )

    object_mask_directory = (
        SOURCE_OBJECT_MASK_DIR
        / mask_directory_name
    )

    annotation_info_path = (
        object_mask_directory
        / "annotation_info.json"
    )

    if not source_image_path.exists():

        raise FileNotFoundError(
            f"Source image does not exist: {source_image_path}"
        )

    if not object_mask_directory.exists():

        raise FileNotFoundError(
            f"Object mask directory does not exist: "
            f"{object_mask_directory}"
        )

    if not annotation_info_path.exists():

        raise FileNotFoundError(
            f"annotation_info.json does not exist: "
            f"{annotation_info_path}"
        )

    image_array = load_source_image(
        source_image_path
    )

    annotation_info = load_json(
        annotation_info_path
    )

    (
        semantic_mask,
        valid_objects,
        invalid_objects
    ) = build_semantic_mask(
        image_shape=image_array.shape,
        annotation_info=annotation_info,
        object_mask_directory=object_mask_directory
    )

    output_image_filename = (
        source_image_path.name
    )

    output_mask_filename = (
        f"{sample_id}.png"
    )

    output_npz_filename = (
        f"{sample_id}.npz"
    )

    output_overlay_filename = (
        f"{sample_id}_overlay.png"
    )

    output_image_path = (
        OUTPUT_IMAGE_DIR
        / output_image_filename
    )

    output_mask_path = (
        OUTPUT_MASK_DIR
        / output_mask_filename
    )

    output_npz_path = (
        OUTPUT_NPZ_DIR
        / output_npz_filename
    )

    output_overlay_path = (
        OUTPUT_OVERLAY_DIR
        / output_overlay_filename
    )

    if COPY_IMAGES:

        copy_source_image(
            source_path=source_image_path,
            destination_path=output_image_path
        )

    save_semantic_mask(
        semantic_mask=semantic_mask,
        output_path=output_mask_path
    )

    if CREATE_NPZ:

        save_npz_file(
            image_array=image_array,
            semantic_mask=semantic_mask,
            output_path=output_npz_path,
            sample_id=sample_id,
            external_id=external_id
        )

    if CREATE_OVERLAYS:

        create_overlay(
            image_array=image_array,
            semantic_mask=semantic_mask,
            class_colors=class_colors,
            output_path=output_overlay_path
        )

    unique_classes = [
        int(class_id)
        for class_id in np.unique(
            semantic_mask
        )
    ]

    foreground_pixels = int(
        np.count_nonzero(
            semantic_mask
        )
    )

    return {
        "sample_id": sample_id,
        "external_id": external_id,
        "source_image": str(
            source_image_path
        ),
        "output_image": (
            output_image_filename
            if COPY_IMAGES
            else None
        ),
        "output_mask": output_mask_filename,
        "output_npz": (
            output_npz_filename
            if CREATE_NPZ
            else None
        ),
        "output_overlay": (
            output_overlay_filename
            if CREATE_OVERLAYS
            else None
        ),
        "image_width": int(
            image_array.shape[1]
        ),
        "image_height": int(
            image_array.shape[0]
        ),
        "foreground_pixels": foreground_pixels,
        "semantic_classes": unique_classes,
        "valid_object_count": len(
            valid_objects
        ),
        "invalid_object_count": len(
            invalid_objects
        ),
        "valid_objects": valid_objects,
        "invalid_objects": invalid_objects,
        "blank_semantic_mask": (
            foreground_pixels == 0
        )
    }


# ==========================================================
# Dataset Processing Pipeline
# ==========================================================

def build_dataset():
    """
    Build the complete semantic segmentation dataset.
    """

    validate_configuration()
    create_output_directories()

    metadata = load_metadata()

    id_to_name, _ = build_class_lookup(
        metadata
    )

    class_colors = build_class_color_lookup(
        metadata
    )

    records = discover_records()

    class_statistics = initialize_class_statistics(
        id_to_name
    )

    processing_report = []

    successful_sample_ids = []

    failed_records = []

    logging.info(
        "Processing %d dataset records.",
        len(records)
    )

    for index, record in enumerate(
            records,
            start=1
    ):

        external_id = record.get(
            "external_id"
        )

        logging.info(
            "Processing %d/%d: %s",
            index,
            len(records),
            external_id
        )

        try:

            result = process_record(
                record=record,
                metadata=metadata,
                class_colors=class_colors
            )

            processing_report.append(
                result
            )

            successful_sample_ids.append(
                result["sample_id"]
            )

            semantic_mask_path = (
                OUTPUT_MASK_DIR
                / result["output_mask"]
            )

            semantic_mask = np.array(
                Image.open(
                    semantic_mask_path
                ),
                dtype=np.uint8
            )

            update_class_statistics(
                statistics=class_statistics,
                semantic_mask=semantic_mask,
                valid_objects=result[
                    "valid_objects"
                ]
            )

            if result["blank_semantic_mask"]:

                logging.warning(
                    "Semantic mask is blank for sample %s.",
                    result["sample_id"]
                )

        except Exception as error:

            logging.exception(
                "Failed to process %s: %s",
                external_id,
                error
            )

            failed_records.append(
                {
                    "external_id": external_id,
                    "mask_directory": record.get(
                        "mask_directory"
                    ),
                    "error": str(error)
                }
            )

    if not successful_sample_ids:

        raise RuntimeError(
            "No dataset samples were processed successfully."
        )

    splits = split_sample_ids(
        successful_sample_ids
    )

    save_split_files(
        splits
    )

    finalize_class_statistics(
        class_statistics
    )

    save_json(
        class_statistics,
        OUTPUT_STATISTICS_PATH
    )

    dataset_metadata = {
        **metadata,
        "source_export_directory": str(
            EXPORT_DIR
        ),
        "output_directory": str(
            OUTPUT_DIR
        ),
        "semantic_mask_format": (
            "single_channel_class_id_png"
        ),
        "npz_fields": [
            "image",
            "mask",
            "sample_id",
            "external_id"
        ],
        "overlap_policy": OVERLAP_POLICY
    }

    save_json(
        dataset_metadata,
        OUTPUT_METADATA_PATH
    )

    report_data = {
        "successful_samples": processing_report,
        "failed_records": failed_records
    }

    save_json(
        report_data,
        OUTPUT_REPORT_PATH
    )

    blank_mask_count = sum(
        1
        for result in processing_report
        if result["blank_semantic_mask"]
    )

    total_valid_objects = sum(
        result["valid_object_count"]
        for result in processing_report
    )

    total_invalid_objects = sum(
        result["invalid_object_count"]
        for result in processing_report
    )

    summary = {
        "source_records": len(records),
        "successful_samples": len(
            successful_sample_ids
        ),
        "failed_samples": len(
            failed_records
        ),
        "blank_semantic_masks": blank_mask_count,
        "valid_object_masks": total_valid_objects,
        "invalid_object_masks": total_invalid_objects,
        "split_sizes": {
            name: len(sample_ids)
            for name, sample_ids in splits.items()
        },
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "random_seed": RANDOM_SEED,
        "overlap_policy": OVERLAP_POLICY,
        "create_npz": CREATE_NPZ,
        "create_overlays": CREATE_OVERLAYS
    }

    save_json(
        summary,
        OUTPUT_SUMMARY_PATH
    )

    save_dataset_yaml(
        metadata=metadata,
        splits=splits
    )

    return summary


# ==========================================================
# Console Summary
# ==========================================================

def print_summary(summary):
    """
    Print a concise dataset build summary.
    """

    print()
    print("=" * 60)
    print("DATASET BUILD COMPLETED")
    print("=" * 60)

    print(
        "Source records:",
        summary["source_records"]
    )

    print(
        "Successful samples:",
        summary["successful_samples"]
    )

    print(
        "Failed samples:",
        summary["failed_samples"]
    )

    print(
        "Blank semantic masks:",
        summary["blank_semantic_masks"]
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
        "Train samples:",
        summary["split_sizes"]["train"]
    )

    print(
        "Validation samples:",
        summary["split_sizes"]["val"]
    )

    print(
        "Test samples:",
        summary["split_sizes"]["test"]
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print("=" * 60)


# ==========================================================
# CLI
# ==========================================================

def parse_arguments():
    """
    Parse optional command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build a semantic segmentation dataset from "
            "individual Labelbox object masks."
        )
    )

    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Do not generate visualization overlays."
    )

    parser.add_argument(
        "--no-npz",
        action="store_true",
        help="Do not generate compressed NPZ files."
    )

    return parser.parse_args()


# ==========================================================
# Main
# ==========================================================

def main():
    """
    Run the dataset builder.
    """

    global CREATE_OVERLAYS
    global CREATE_NPZ

    arguments = parse_arguments()

    if arguments.no_overlays:
        CREATE_OVERLAYS = False

    if arguments.no_npz:
        CREATE_NPZ = False

    summary = build_dataset()

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()