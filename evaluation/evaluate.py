"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-evaluation-test-phase | gzip -c > example-evaluation-test-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method('spawn', force=True)
import json
from glob import glob
import SimpleITK
import numpy as np
import random
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint
from image_metrics import ImageMetrics
from segmentation_metrics import SegmentationMetrics
import gc
import torch
from functools import partial
import os

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

def tree(dir_path: Path, prefix: str=''):

    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """    
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)


def init_pool(image_evaluator, segmentation_evaluator, debug, has_predictions):
    """Initializer to set global variables in worker processes."""
    # this is a bit hacky but it works. Basically we load every evaluator
    # once and put them in a global variable. Then each spawned process can access these 
    # variables. This means we only load our pytorch model once
    global _image_evaluator, _segmentation_evaluator, _debug, _has_predictions
    _image_evaluator = image_evaluator
    _segmentation_evaluator = segmentation_evaluator
    _debug = debug
    _has_predictions = has_predictions

def main():
    _debug = 'DEBUG' in os.environ and str(os.environ['DEBUG']).lower() in ['1', 'true', 't']
    nprocs = int(os.environ['NPROCS']) if 'NPROCS' in os.environ and int(os.environ['NPROCS']) > 0 else 2

    if _debug:
        print("INPUT DIR")
        for line in tree(INPUT_DIRECTORY):
            print(line)

        print("")

        print("OUTPUT DIR")
        for line in tree(OUTPUT_DIRECTORY):
            print(line)

        print("")
        print("GT DIR")
        for line in tree(GROUND_TRUTH_DIRECTORY):
            print(line)


        if os.path.isfile(INPUT_DIRECTORY / "inputs.json"):
            print("")
            print("Found inputs.json. Contents: ")
            with open(INPUT_DIRECTORY / "inputs.json", 'r') as f:
                j = json.loads(f.read())

            print(json.dumps(j, indent=2))
            print("")

    print(f"Running {nprocs} process{'' if nprocs==1 else 'es'}")

    metrics = {}
    if os.path.isfile(INPUT_DIRECTORY / "predictions.json"):
        predictions = read_predictions()
        has_predictions = True
    else:
        print(f"We're in a prediction-only phase. Reading files directly from {INPUT_DIRECTORY}")
        predictions = glob(str(INPUT_DIRECTORY / "*.mha"))
        has_predictions = False


    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon
    # global _image_evaluator, _segmentation_evaluator
    _image_evaluator = ImageMetrics(debug=_debug)

    _segmentation_evaluator = SegmentationMetrics(debug=_debug)

    metrics['results'] = []

    with torch.multiprocessing.Pool(processes=nprocs, initializer=init_pool, initargs=(_image_evaluator, _segmentation_evaluator, _debug, has_predictions)) as pool:
        try:
            metrics["results"] = pool.map(process, predictions)
        except KeyboardInterrupt:
            print('Caught Ctrl+C, shutting pool down...')
            pool.terminate()
            pool.join()

    if _debug:
        print(metrics)

    # Now generate an overall score(s) for this submission. 
    # For every case in the dataset, we have a metric. For each metric,
    # The aggregates listed below are computed over the entire dataset
    aggregate_functions = [
        {
            'name': 'mean',
            'function': np.mean
        },
        {
            'name': 'max',
            'function': np.max
        },
        {
            'name': 'min',
            'function': np.min
        },
        {
            'name': 'std',
            'function': np.std
        },
        {
            'name': '25pc',
            'function': partial(np.quantile, q=0.25)
        },
        {
            'name': '50pc',
            'function': partial(np.quantile, q=0.50)
        },
        {
            'name': '75pc',
            'function': partial(np.quantile, q=0.75)
        },
        {
            'name': 'count',
            'function': len
        },

    ]
    metrics["aggregates"] = {}
    

    if len(metrics['results']) > 0:
        for metric in metrics["results"][0].keys():
            metrics["aggregates"][metric] = {}
            all_results = [result[metric] for result in metrics["results"]]
            for aggregate_function in aggregate_functions:
                metrics["aggregates"][metric][aggregate_function['name']] = aggregate_function['function'](all_results)


    if _debug:
        print(metrics)

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0

def process(job):
    # Processes a single algorithm job, looking at the outputs
    gc.collect()
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    if _has_predictions:
        synthetic_ct_location = get_file_location(
                job_pk=job["pk"],
                values=job["outputs"],
                slug="synthetic-ct-image",
            )
    else:
        synthetic_ct_location = job

    # Extract the patient ID
    if _has_predictions:
        patient_id = find_patient_id(values=job["inputs"], slug='body')
    else:
        # filename is like "/input/sct_1HNXxxx.mha"
        patient_id = job.split('/')[-1].split('.')[0][-7:]

    # and load the ground-truth along the affine image matrix (Or direction/origin/spacing in SimpleITK terms)
    gt_img, spacing, origin, direction = load_image_file_directly(location=GROUND_TRUTH_DIRECTORY / "ct" / f"{patient_id}.mha", return_orientation=True)

    if _has_predictions:
        # Then, read the sCT and impose the spatial dimension of the ground truth
        synthetic_ct, full_sct_path = load_image_file(
            location=synthetic_ct_location, spacing=spacing, origin=origin, direction=direction
        )
    else:
        synthetic_ct = load_image_file_directly(location=synthetic_ct_location, set_orientation=(spacing, origin, direction))
        full_sct_path = synthetic_ct_location

    # Do the same for the ground-truth TotalSegmentator segmentation and the mask
    gt_segmentation = load_image_file_directly(location=GROUND_TRUTH_DIRECTORY / "segmentation" / f"{patient_id}.mha", set_orientation=(spacing, origin, direction))

    mask = load_image_file_directly(location=GROUND_TRUTH_DIRECTORY / "mask" / f"{patient_id}.mha", set_orientation=(spacing, origin, direction))

    # score the subject based on image metrics
    image_metrics = _image_evaluator.score_patient(gt_img, synthetic_ct, mask)

    gc.collect()
    #... and segmentation metrics
    seg_metrics = _segmentation_evaluator.score_patient(full_sct_path, mask, gt_segmentation, patient_id, orientation=(spacing, origin, direction))

    if _debug:
        print(patient_id, {**image_metrics, **seg_metrics})
    # Finally, return the results
    gc.collect()
    return {
        **image_metrics,
        **seg_metrics
    }

def find_patient_id(*, values, slug):
    # find the patient id (e.g. TXXXYYY, where T is task (1 or 2), XXX is anatomy and center 
    # (e.g., THC for thorax from center C) and YYY is the patient number (e.g., 001))
    for value in values:
        if value["interface"]["slug"] == slug:
            full_name = value['image']['name'] # this name is like "mask_1ABCxxx.mha"
            return full_name.split('.')[0].split('_')[-1]
    raise RuntimeError(f"Cannot get patient name because interface {slug} not found!")

def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*.mha") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")

def get_input_file_location(*, values, slug):
    relative_path = get_interface_relative_path(values=values, slug=slug)
    for value in values:
        if value["interface"]["slug"] == slug:
            full_name = value['image']['name'] # this name is like "mask_1ABCxxx.mha"

            return INPUT_DIRECTORY / relative_path / full_name

    raise RuntimeError(f"Cannot find input file for {slug}!")

def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_image_file_directly(*, location, return_orientation=False, set_orientation=None):
    # immediatly load the file and find its orientation
    result = SimpleITK.ReadImage(location)
    # Note, transpose needed because Numpy is ZYX according to SimpleITKs XYZ
    img_arr = np.transpose(SimpleITK.GetArrayFromImage(result), [2, 1, 0])

    if return_orientation:
        spacing = result.GetSpacing()
        origin = result.GetOrigin()
        direction = result.GetDirection()


        return img_arr, spacing, origin, direction
    else:
        # If desired, force the orientation on an image before converting to NumPy array
        if set_orientation is not None:
            spacing, origin, direction = set_orientation
            result.SetSpacing(spacing)
            result.SetOrigin(origin)
            result.SetDirection(direction)

        # Note, transpose needed because Numpy is ZYX according to SimpleITKs XYZ
        return np.transpose(SimpleITK.GetArrayFromImage(result), [2, 1, 0])


def load_image_file(*, location, spacing=None, origin=None, direction=None):
    # Use SimpleITK to read a file in a directory
    input_files = glob(str(location / "*.nii.gz")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    if spacing is not None:
        result.SetSpacing(spacing)
    if origin is not None:
        result.SetOrigin(origin)
    if direction is not None:
        result.SetDirection(direction)

    # Convert it to a Numpy array
    return np.transpose(SimpleITK.GetArrayFromImage(result), [2, 1, 0]), input_files[0]


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
