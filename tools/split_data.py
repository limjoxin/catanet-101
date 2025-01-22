import argparse
import os
import glob
import shutil
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='path to folder containing processed video-folders.'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='../output',
        help='path to output folder.'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='Split train and validation for 6-fold cross validation. Select which fold to split for.'
    )
    args = parser.parse_args()
    
    # Look for case folders inside phase_recognition directory
    videos = glob.glob(os.path.join(args.input, 'phase_recognition/case_*'))

    if not os.path.isdir(args.out):
        [os.makedirs(os.path.join(args.out, phase)) for phase in ['train', 'val', 'test']]

    with open('data_config.yaml', 'r') as ymlfile:
        dataconfig = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    ids_phase = {}
    ids_phase['train'] = []
    for fold in dataconfig['train']:
        if fold != f'fold{args.fold}':
            ids_phase['train'] += dataconfig['train'][fold]
    ids_phase['val'] = dataconfig['train'][f'fold{args.fold}']
    ids_phase['test'] = dataconfig['test']

    for phase in ids_phase:
        for id in ids_phase[phase]:
            # Look for the case folder in the phase_recognition directory
            filepath = os.path.join(args.input, 'phase_recognition', f'case_{id}')
            
            if not os.path.exists(filepath):
                assert False, f'folder case_{id} not found in {os.path.join(args.input, "phase_recognition")}'
            
            destination = os.path.join(args.out, phase, f'case_{id}')
            print(f"  Moving to: {destination}")
            shutil.move(filepath, destination)


