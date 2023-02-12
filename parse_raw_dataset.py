import argparse
import json
import os
import random
import shutil
import uuid
from typing import Optional

from PIL import Image, UnidentifiedImageError


def _clear_string(raw_string):
    if raw_string[0] == "\ufeff":
        raw_string = raw_string[1:]
    if raw_string[-1] == "\xa0":
        raw_string = raw_string[:-1]
    return raw_string


def parse_raw_data(
        source_path: str,
        result_path: str,
        description_file_name: str,
        dataset_structure_file_name: str,
        archive_result_path: Optional[str] = None,
        remove_result_folder: bool = False,
) -> dict:
    print(
        f"parse_raw_data with next args: \n"
        f"source_path: {source_path}\n"
        f"result_path: {result_path}\n"
        f"description_file_name: {description_file_name}\n"
        f"dataset_structure_file_name: {dataset_structure_file_name}\n"
        f"archive_result_path: {archive_result_path}\n"
        f"remove_result_folder: {remove_result_folder}\n"
    )
    # clear result folder
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    item_number = 0
    dataset_structure = dict()
    statistics = {
        "general": dict(),
        "details": dict(),
    }
    category_dir: os.DirEntry
    # main loop
    for category_dir in os.scandir(source_path):
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name
        statistics["details"][category_name] = dict()
        statistics["general"][category_name] = 0
        # handle brands
        brand_dir: os.DirEntry
        for brand_dir in os.scandir(os.path.join(source_path, category_name)):
            if not brand_dir.is_dir():
                continue
            brand_name = brand_dir.name
            statistics["details"][category_name][brand_name] = 0
            # handle model name
            model_dir: os.DirEntry
            for model_dir in os.scandir(brand_dir.path):
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                model_dir_path = os.path.join(source_path, category_name, brand_name, model_name)
                result_model_dir_path = os.path.join(
                    result_path,
                    category_name,
                    brand_name,
                    model_name,
                )
                os.makedirs(result_model_dir_path, exist_ok=True)
                statistics["details"][category_name][brand_name] += 1
                # get description
                description_file_path = os.path.join(model_dir_path, description_file_name)
                if not os.path.exists(description_file_path):
                    print(f"Can't find description file at path: {description_file_path}")
                    raise ValueError
                with open(description_file_path, "r", encoding="utf-8") as desc_file:
                    try:
                        raw_text = desc_file.read().splitlines()
                    except UnicodeDecodeError:
                        raw_text = list()
                    # clear
                    text = [_clear_string(t) for t in raw_text if len(t) > 10]
                # handle files images
                file: os.DirEntry
                for file in os.scandir(model_dir_path):
                    if file.is_dir():
                        continue
                    # handle images
                    try:
                        image = Image.open(file.path)  # type: Image.Image
                    except UnidentifiedImageError:
                        continue
                    # save to jpg with uuid name
                    file_name = f"{uuid.uuid4()}.jpg"
                    image_result_path = os.path.join(result_model_dir_path, file_name)
                    rgb_image = image.convert("RGB")
                    rgb_image.save(image_result_path)
                    # add dataset_structure node
                    dataset_structure[item_number] = {
                        "category": category_name,
                        "brand": brand_name,
                        "model": model_name,
                        "img": file_name,
                        "text": text,
                    }
                    item_number += 1
                    statistics["general"][category_name] += 1
    # save dataset_structure to file
    print(f"Total {item_number} image collected.")
    print(f"Statistics: {json.dumps(statistics, indent=2)}")
    with open(os.path.join(result_path, dataset_structure_file_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_structure, ensure_ascii=False, indent=2))
    # archive if required
    if archive_result_path is not None:
        result_archive_file_path = os.path.join(archive_result_path, "matryoshka")
        shutil.make_archive(result_archive_file_path, "zip", result_path)
        print(f"Result was archived to {result_archive_file_path}.zip")
    # cleaning
    if remove_result_folder:
        shutil.rmtree(result_path)
        print(f"result path: {result_path} was cleared")
    return dataset_structure


def _save_new_files(
        numbers: list[int],
        dataset_structure: dict,
        result_path: str,
        dataset_structure_file_name: str,
        postfix: str,
) -> dict:
    new_result_path = f"{result_path}_{postfix}"
    if os.path.exists(new_result_path):
        shutil.rmtree(new_result_path)
    os.makedirs(new_result_path, exist_ok=True)
    new_dataset_structure = dict()
    for number in numbers:
        node_info = dataset_structure[number]
        new_dataset_structure[number] = node_info
        new_result_folder_path = os.path.join(
            new_result_path,
            node_info["category"],
            node_info["brand"],
            node_info["model"],
        )
        origin_file_path = os.path.join(
            result_path,
            node_info["category"],
            node_info["brand"],
            node_info["model"],
            node_info["img"],
        )
        os.makedirs(new_result_folder_path, exist_ok=True)
        shutil.copyfile(origin_file_path, os.path.join(new_result_folder_path, node_info["img"]))
    print(f"Total {postfix} len: {len(new_dataset_structure)}")
    with open(os.path.join(new_result_path, dataset_structure_file_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(new_dataset_structure, ensure_ascii=False, indent=2))
    return new_dataset_structure


def train_test_split(
        source_path: str,
        result_path: str,
        description_file_name: str,
        dataset_structure_file_name: str,
        archive_result_path: Optional[str] = None,
        remove_result_folder: bool = False,
        train_share: Optional[float] = 0.8,
        seed: Optional[int] = 42,
):
    dataset_structure = parse_raw_data(
        source_path=source_path,
        result_path=result_path,
        description_file_name=description_file_name,
        dataset_structure_file_name=dataset_structure_file_name,
        archive_result_path=archive_result_path,
        remove_result_folder=False,
    )
    numbers_by_category = dict()
    for number, data in dataset_structure.items():
        category_name = data["category"]
        try:
            numbers_by_category[category_name].append(number)
        except KeyError:
            numbers_by_category[category_name] = [number]
    train_numbers = list()
    test_numbers = list()
    for category_name, numbers in numbers_by_category.items():
        random.seed(seed)
        random.shuffle(numbers)
        border_idx = int(len(numbers) * train_share)
        train_numbers.extend(numbers[:border_idx])
        test_numbers.extend(numbers[border_idx:])
    _save_new_files(train_numbers, dataset_structure, result_path, dataset_structure_file_name, postfix="train")
    _save_new_files(test_numbers, dataset_structure, result_path, dataset_structure_file_name, postfix="test")
    # cleaning
    if remove_result_folder:
        shutil.rmtree(result_path)
        print(f"result path: {result_path} was cleared")
    # TODO archive result folder if necessary


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Process some params.")
    parser.add_argument(
        "--source_path",
        action="store",
        dest="source_path",
        help="Source folder path",
        type=str,
    )
    parser.add_argument(
        "--result_path",
        action="store",
        dest="result_path",
        help="Result path",
        type=str,
    )
    parser.add_argument(
        "--description_file_name",
        action="store",
        dest="description_file_name",
        help="Description file name",
        type=str,
        default="desc.txt",
    )
    parser.add_argument(
        "--dataset_structure_file_name",
        action="store",
        dest="dataset_structure_file_name",
        help="Dataset structure file name",
        type=str,
        default="matryoshka.json",
    )
    parser.add_argument(
        "--archive_result_path",
        action="store",
        dest="archive_result_path",
        help="Archive result path without file name",
        type=str,
    )
    parser.add_argument(
        "--remove_result_folder",
        action="store_true",
        dest="remove_result_folder",
        help="Remove result folder.",
    )
    parser_args = parser.parse_args()
    # TODO rewrite on classes
    train_test_split(
        source_path=parser_args.source_path or os.path.join("matryoshka_raw"),
        result_path=parser_args.result_path or os.path.join("matryoshka"),
        description_file_name=parser_args.description_file_name,
        dataset_structure_file_name=parser_args.dataset_structure_file_name,
        archive_result_path=parser_args.archive_result_path,
        remove_result_folder=parser_args.remove_result_folder,
    )
