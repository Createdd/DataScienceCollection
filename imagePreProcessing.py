import os
import glob
import shutil
import typing
import numpy as np
from PIL import Image, ImageStat, ImageChops
from pathlib import Path


def pre_processsing(input_dir: str, output_dir: str, logfile: str) -> int:
    """Preprocess images for further ML development

    Args:
        input_dir (str): Directory with images
        output_dir (str): Directory to copy valid images
        logfile (str): Logfile for invalid images

    Returns:
        int: Number of valid files
    """
    sorted_files = get_files(input_dir)

    checked_files = check_files(sorted_files, output_dir)

    valid_files, invalid_files = separate_files(checked_files)

    copy_valid_files(valid_files, output_dir)

    write_log_file(logfile, invalid_files, input_dir)

    return len(valid_files)


def get_files(input_dir: str) -> typing.List[str]:
    """Get files of directory recursively and sort them

    Args:
        input_dir (str): [description]

    Returns:
        typing.List[str]: [description]
    """
    sorted_files = []
    sorted_files.extend([file for file in glob.glob(
        os.path.join(input_dir, f'**/*.*'), recursive=True)])
    sorted_files = sorted(sorted_files)
    return sorted_files


def check_files(sorted_files: typing.List[str], output_dir: str) -> typing.List[typing.Union[str, int]]:
    """Validate files for certain criteria needed for images

    Args:
        sorted_files (typing.List[str]): [description]
        output_dir (str): [description]

    Returns:
        typing.List[typing.Union[str, int]]: List of images files and corresponding number of error
    """
    checked_files = []
    for i, file_name in enumerate(sorted_files):
        nr_of_error = validate_file(file_name, output_dir, checked_files)
        checked_files.append([file_name, nr_of_error])

    return checked_files


def validate_file(file_path: str, output_dir: str, checked_files: typing.List[str]) -> int:
    """Valid image for certain errors

    Args:
        file_path (str): File path for
        output_dir (str): [description]
        checked_files (typing.List[str]): [description]

    Returns:
        int: Number of error
    """
    nr_of_error = 0

    # error 3
    try:
        im = Image.open(file_path)
    except IOError:
        nr_of_error = 3
        return nr_of_error

    # error 1
    allowed_extensions = ('.jpg', '.JPG', '.JPEG', '.jpeg')
    file_has_allowed_extension = file_path.lower().endswith(allowed_extensions)
    if not file_has_allowed_extension:
        nr_of_error = 1
        return nr_of_error

    # error 2
    file_size = os.path.getsize(file_path)
    file_has_allowed_size = file_size > 10000
    if not file_has_allowed_size:
        nr_of_error = 2
        return nr_of_error

    # error 4
    img = Image.open(file_path)
    variance = ImageStat.Stat(img)
    file_has_variance = sum(variance.var) > 0
    if not file_has_variance:
        nr_of_error = 4
        return nr_of_error

    # error 5
    im = Image.open(file_path)
    width, height = im.size
    file_has_dims = width >= 100 and height >= 100
    shape = np.array(img).shape
    if not file_has_dims or len(shape) > 2:
        nr_of_error = 5
        return nr_of_error

    # error 6
    def is_equal(im1, im2):
        """Compare images to check if they are the same

        Args:
            im1 ([type]): [description]
            im2 ([type]): [description]

        Returns:
            [type]: Difference between images or None. If the comparison fails return the error number.
        """
        try:
            res = ImageChops.difference(im1, im2).getbbox()
            return res
        except ValueError:
            res = 6
            return res

    im1 = Image.open(file_path)
    for image in checked_files:
        image_path = image[0]
        if image[1] != 3 and image[1] != 1:
            im2 = Image.open(image_path)
            res = is_equal(im1, im2)
            if res == None:
                nr_of_error = 6
                return nr_of_error

    return nr_of_error


def separate_files(checked_files: typing.List[typing.Union[str, int]]):
    """Separate the checked files into valid and invalid image files

    Args:
        checked_files (typing.List[typing.Union[str, int]]): [description]

    Returns:
        [type]: List of valid and invalid files
    """
    valid_files = []
    invalid_files = []
    for checked in checked_files:
        if checked[1] == 0:
            valid_files.append(checked)
        else:
            invalid_files.append(checked)

    return valid_files, invalid_files


def copy_valid_files(valid_files: typing.List[str], output_dir: str):
    """Copy the valid files into a desired output directory

    Args:
        valid_files (typing.List[str]): [description]
        output_dir (str): [description]
    """
    for i, elem in enumerate(valid_files):
        source = elem[0]
        destination = os.path.join(output_dir, f'{i+1:06d}.jpg')
        shutil.copyfile(source, destination)


def write_log_file(logfile: str, invalid_files: typing.List[str], input_dir: str):
    """Writes a log file to inspect the invalid files and their errors

    Args:
        logfile (str): [description]
        invalid_files (typing.List[str]): [description]
        input_dir (str): [description]
    """
    with open(logfile, 'w') as f:
        for file_name in invalid_files:
            realtive_path = Path(file_name[0]).relative_to(
                os.path.abspath(input_dir))
            f.write(f'{realtive_path};{file_name[1]}\n')
        f.close()
