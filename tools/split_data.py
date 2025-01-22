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
    
    # Print input directory and check if it exists
    print(f"\nDEBUG: Input directory path: {args.input}")
    print(f"DEBUG: Input directory exists: {os.path.exists(args.input)}")
    
    # Look for case folders inside phase_recognition directory
    videos = glob.glob(os.path.join(args.input, 'case_*'))
    print(f"\nDEBUG: Found {len(videos)} video folders")
    print("DEBUG: First few video folders found:", videos[:5])
    
    if not os.path.isdir(args.out):
        [os.makedirs(os.path.join(args.out, phase)) for phase in ['train', 'val', 'test']]
        print(f"\nDEBUG: Created output directories in {args.out}")

    print("\nDEBUG: Loading data_config.yaml...")
    try:
        with open('data_config.yaml', 'r') as ymlfile:
            dataconfig = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            print("DEBUG: Successfully loaded data_config.yaml")
    except Exception as e:
        print(f"DEBUG: Error loading data_config.yaml: {str(e)}")
        raise

    ids_phase = {}
    ids_phase['train'] = []
    
    print(f"\nDEBUG: Processing fold {args.fold}")
    for fold in dataconfig['train']:
        if fold != f'fold{args.fold}':
            print(f"DEBUG: Adding cases from {fold}")
            ids_phase['train'] += dataconfig['train'][fold]
    
    print(f"DEBUG: Total training cases: {len(ids_phase['train'])}")
    
    ids_phase['val'] = dataconfig['train'][f'fold{args.fold}']
    print(f"DEBUG: Validation cases: {len(ids_phase['val'])}")
    print(f"DEBUG: First few validation cases: {ids_phase['val'][:5]}")
    
    ids_phase['test'] = dataconfig['test']
    print(f"DEBUG: Test cases: {len(ids_phase['test'])}")

    # Create a set of available case IDs for quick lookup
    available_cases = {os.path.basename(v).replace('case_', '') for v in videos}
    print("\nDEBUG: Available case IDs:", sorted(list(available_cases))[:10], "...")

    for phase in ids_phase:
        print(f"\nDEBUG: Processing {phase} phase")
        for id in ids_phase[phase]:
            # Look for the case folder in the phase_recognition directory
            filepath = os.path.join(args.input, f'case_{id}')
            print(f"\nDEBUG: Looking for case_{id}")
            print(f"DEBUG: Full path being checked: {filepath}")
            print(f"DEBUG: Path exists: {os.path.exists(filepath)}")
            
            if not os.path.exists(filepath):
                print(f"DEBUG: Available files in {args.input}:")
                try:
                    print(os.listdir(args.input)[:10])
                except Exception as e:
                    print(f"DEBUG: Error listing directory: {str(e)}")
                assert False, f'folder case_{id} not found in {os.path.join(args.input)}'
            
            destination = os.path.join(args.out, phase, f'case_{id}')
            print(f"DEBUG: Moving {filepath} to {destination}")
            shutil.move(filepath, destination)
