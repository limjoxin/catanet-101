import os
import glob
import subprocess
import cv2
import pandas as pd
import numpy as np
import argparse
import warnings
import tempfile

# crop videos to labelled start and end of surgery
# resize to 256x256 to speed-up data loading and processing
# extract frames at original video frame rate to preserve all data for stratified sampling

def frame2minsec(frame, fps):
    """Convert frame number to MM:SS format based on fps."""
    seconds = frame/fps
    minutes = np.floor(seconds / 60).astype(int)
    seconds = np.floor(seconds % 60).astype(int)
    minsec = '{0:02d}:{1:02d}'.format(minutes, seconds)
    return minsec

def extract_frames_from_video(v_path, out_path, start, end, dim=256, use_original_fps=True):
    '''Extract frames from a video at original framerate or specified fps.
    
    v_path: single video path
    out_path: root to store output frames
    start: start time in MM:SS format
    end: end time in MM:SS format
    dim: dimension to resize frames to (square)
    use_original_fps: if True, extract all frames; if False, sample at 2.5fps
    '''
    vname = os.path.splitext(os.path.basename(v_path))[0]
    vidcap = cv2.VideoCapture(v_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    orig_fps = vidcap.get(cv2.CAP_PROP_FPS)

    if (width == 0) or (height==0):
        print(v_path, 'not successfully loaded, drop ..'); return

    start = '00:'+ start
    end = '00:' + end

    if not os.path.isdir(os.path.join(out_path, vname)):
        os.makedirs(os.path.join(out_path, vname))
    tmpfolder = tempfile.gettempdir()
    
    # Step 1: Resize + crop video to the specified time range
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-i', '%s'% v_path,
           '-ss', start,
           '-to', end,
           '-vf',
           'scale={0:d}:{0:d}'.format(dim),
           '%s' % os.path.join(tmpfolder, vname + '.mp4')]
    ffmpeg = subprocess.call(cmd)
    
    # Step 2: Extract frames - either at original fps or at specified fps
    if use_original_fps:
        # Extract all frames (no fps filter)
        print(f"Extracting frames at original frame rate ({orig_fps:.2f} fps)")
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
               '-i', '%s' % os.path.join(tmpfolder, vname + '.mp4'),
               '-vsync', '0',  # Maintain exact frame timestamps
               '%s' % os.path.join(out_path, vname, vname + '_%04d.png')]
    else:
        # Extract at reduced fps (e.g., 2.5 fps)
        target_fps = 2.5
        print(f"Extracting frames at reduced rate ({target_fps} fps)")
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
               '-i', '%s' % os.path.join(tmpfolder, vname + '.mp4'),
               '-vf',
               'fps={0:.1f}'.format(target_fps),
               '%s' % os.path.join(out_path, vname, vname + '_%04d.png')]
    
    ffmpeg = subprocess.call(cmd)


def find_case_directories(input_path):
    """Find all case directories within the input path."""
    return [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]


def main_cataracts_train(input_path, output_path, use_original_fps=True):
    """Process videos from the CATARACTs training dataset at original frame rate.
    Videos and CSVs are organized in case_id subdirectories."""
    print('save to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Find all case directories
    case_dirs = find_case_directories(input_path)
    print(f"Found {len(case_dirs)} case directories")
    
    for case_dir in case_dirs:
        case_path = os.path.join(input_path, case_dir)
        print(f"Processing case directory: {case_dir}")
        
        # Find videos and labels in this case directory
        videos = sorted(glob.glob(os.path.join(case_path, '*.mp4')))
        labels = sorted(glob.glob(os.path.join(case_path, '*.csv')))
        
        if not videos:
            print(f"No MP4 videos found in {case_path}, skipping...")
            continue
            
        if len(videos) != len(labels):
            print(f"Warning: Not same number of videos ({len(videos)}) and labels ({len(labels)}) in {case_path}")
            continue
        
        for i, (vid, lab) in enumerate(zip(videos, labels)):
            print(f'{vid}: {i+1} of {len(videos)}')
            vname = os.path.splitext(os.path.basename(vid))[0]
            
            if os.path.isdir(os.path.join(output_path, vname)):
                print(f"Directory for {vname} already exists, skipping...")
                continue
                
            # Find surgery start and end from labels
            label = pd.read_csv(lab)['Steps'].to_numpy()
            label = np.concatenate((label, [0]))  # make sure the last value is 0 to find the edge
            start_frame = np.min(np.where((np.diff(label, prepend=0) != 0) & (label == 3))[0])
            end_frame = np.max(np.where((np.diff(label, prepend=0) != 0) & (label == 0))[0])
            
            # Get original frame rate and convert to timestamps
            dummy_reader = cv2.VideoCapture(vid)
            orig_fps = dummy_reader.get(cv2.CAP_PROP_FPS)
            start = frame2minsec(start_frame, orig_fps)
            end = frame2minsec(end_frame, orig_fps)
            
            # Extract frames
            extract_frames_from_video(vid, output_path, dim=256, 
                                      start=start, end=end, use_original_fps=use_original_fps)


def main_others(input_path, output_path, use_original_fps=True):
    """Process videos with a simple start/end label file at original frame rate.
    Videos are organized in case_id subdirectories."""
    print('save to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Find all case directories
    case_dirs = find_case_directories(input_path)
    print(f"Found {len(case_dirs)} case directories")
    
    # First check if there's a global start_end_labels.csv file
    global_label_path = os.path.join(input_path, 'start_end_labels.csv')
    if os.path.isfile(global_label_path):
        labels = pd.read_csv(global_label_path)
        using_global_labels = True
        print("Using global labels file.")
    else:
        using_global_labels = False
        print("No global labels file found. Will look for labels in each case directory.")
    
    # Process each case directory
    for case_dir in case_dirs:
        case_path = os.path.join(input_path, case_dir)
        print(f"Processing case directory: {case_dir}")
        
        # Find videos in this case directory
        videos = sorted(glob.glob(os.path.join(case_path, '*.mp4')))
        
        if not videos:
            print(f"No MP4 videos found in {case_path}, skipping...")
            continue
        
        # If not using global labels, look for a label file in the case directory
        if not using_global_labels:
            local_label_path = os.path.join(case_path, 'start_end_labels.csv')
            if not os.path.isfile(local_label_path):
                print(f"No start_end_labels.csv found in {case_path}, skipping...")
                continue
            labels = pd.read_csv(local_label_path)
        
        for i, vid in enumerate(videos):
            print(f'{vid}: {i+1} of {len(videos)}')
            vname = os.path.splitext(os.path.basename(vid))[0]
            
            if os.path.isdir(os.path.join(output_path, vname)):
                print(f"Directory for {vname} already exists, skipping...")
                continue
                
            # Get start and end times from labels
            p_label = labels[labels.PatientID.eq(vname)]
            if len(p_label) != 1:
                print(f'No or multiple entries for {vname}, skipping...')
                continue
            
            # Extract frames
            extract_frames_from_video(vid, output_path, dim=256, 
                                      start=p_label['Start'].values[0], 
                                      end=p_label['End'].values[0],
                                      use_original_fps=use_original_fps)


def main_cataract101(input_path, output_path, use_original_fps=True):
    """Process videos from the Cataract-101 dataset at original frame rate.
    Videos and CSVs are organized in case_id subdirectories."""
    print('save to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Find all case directories
    case_dirs = find_case_directories(input_path)
    print(f"Found {len(case_dirs)} case directories")
    
    for case_dir in case_dirs:
        case_path = os.path.join(input_path, case_dir)
        print(f"Processing case directory: {case_dir}")
        
        # Find videos and labels in this case directory
        videos = sorted(glob.glob(os.path.join(case_path, '*.mp4')))
        labels = sorted(glob.glob(os.path.join(case_path, '*.csv')))
        
        if not videos:
            print(f"No MP4 videos found in {case_path}, skipping...")
            continue
            
        if len(videos) != len(labels):
            print(f"Warning: Not same number of videos ({len(videos)}) and labels ({len(labels)}) in {case_path}")
            continue
        
        for i, (vid, lab) in enumerate(zip(videos, labels)):
            print(f'{vid}: {i + 1} of {len(videos)}')
            vname = os.path.splitext(os.path.basename(vid))[0]
            
            if os.path.isdir(os.path.join(output_path, vname)):
                print(f"Directory for {vname} already exists, skipping...")
                continue
            
            # Get original frame rate
            vidcap = cv2.VideoCapture(vid)
            orig_fps = vidcap.get(cv2.CAP_PROP_FPS)
            
            # Process label file to find surgery start and end
            try:
                label_df = pd.read_csv(lab)
                label = np.concatenate((label_df['valid'].to_numpy(), [0]))
                start_frame = np.min(np.where((np.diff(label, prepend=0) != 0) & (label == 1))[0])
                end_frame = np.max(np.where((np.diff(label, prepend=0) != 0) & (label == 0))[0])
            except Exception as e:
                print(f"Error processing label file {lab}: {e}, skipping...")
                continue
            
            # Convert frame numbers to timestamps
            start = frame2minsec(start_frame, orig_fps)
            end = frame2minsec(end_frame, orig_fps)
            
            # Create output directory
            if not os.path.isdir(os.path.join(output_path, vname)):
                os.makedirs(os.path.join(output_path, vname))
                
            # Extract frames
            extract_frames_from_video(vid, output_path, dim=256, 
                                      start=start, end=end, 
                                      use_original_fps=use_original_fps)


def main_cataract1k(input_path, output_path, use_original_fps=True):
    """Process videos from the Cataract-1K Phase Recognition dataset at original frame rate.
    Videos and CSVs are organized in case_id subdirectories."""
    print('Processing Cataract-1K dataset, saving to %s ... ' % output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Find all case directories
    case_dirs = find_case_directories(input_path)
    print(f"Found {len(case_dirs)} case directories")
    
    for case_dir in case_dirs:
        case_path = os.path.join(input_path, case_dir)
        print(f"Processing case directory: {case_dir}")
        
        # Find videos in this case directory
        videos = sorted(glob.glob(os.path.join(case_path, '*.mp4')))
        
        if not videos:
            print(f"No MP4 videos found in {case_path}, skipping...")
            continue
        
        for i, vid in enumerate(videos):
            print(f'Processing {vid}: {i+1} of {len(videos)}')
            
            # Get case ID from the filename or directory
            vname = os.path.splitext(os.path.basename(vid))[0]
            case_id = case_dir  # Assuming directory name is the case_id
            
            # Look for annotation files in the case directory
            phase_csv = os.path.join(case_path, f"case_{case_id}_annotations_phases.csv")
            metadata_csv = os.path.join(case_path, f"case_{case_id}_video.csv")
            
            # If file naming doesn't match directory structure, try extracting from filename
            if not os.path.exists(phase_csv):
                try:
                    case_id_from_filename = vname.split('_')[1]
                    phase_csv = os.path.join(case_path, f"case_{case_id_from_filename}_annotations_phases.csv")
                    metadata_csv = os.path.join(case_path, f"case_{case_id_from_filename}_video.csv")
                except:
                    pass
            
            if os.path.isdir(os.path.join(output_path, vname)):
                print(f"Directory for {vname} already exists, skipping...")
                continue
                
            if not (os.path.exists(phase_csv) and os.path.exists(metadata_csv)):
                print(f"Warning: Missing annotation files for {vname}, skipping...")
                continue
            
            # Read metadata and annotations
            try:
                metadata = pd.read_csv(metadata_csv)
                orig_fps = metadata['fps'].iloc[0]
                phases = pd.read_csv(phase_csv)
                
                # Calculate start and end frames
                start_frame = phases['frame'].min()
                end_frame = phases['endFrame'].max()
            except Exception as e:
                print(f"Error processing annotation files for {vname}: {e}, skipping...")
                continue
            
            # Convert to timestamps
            start = frame2minsec(start_frame, orig_fps)
            end = frame2minsec(end_frame, orig_fps)
            
            # Create output directory
            if not os.path.isdir(os.path.join(output_path, vname)):
                os.makedirs(os.path.join(output_path, vname))
                
            # Extract frames
            extract_frames_from_video(vid, output_path, dim=256, 
                                     start=start, end=end,
                                     use_original_fps=use_original_fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='path to folder containing case subdirectories with video files.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../output',
        help='path to output folder.'
    )
    parser.add_argument(
        '--label',
        type=str,
        choices=['cataract101', 'CATARACTs', 'startend', 'cataract1k'],
        default='cataract101',
        help='type of label dataset'
    )
    parser.add_argument(
        '--original_fps',
        action='store_true',
        help='extract frames at original video framerate instead of 2.5 fps'
    )
    args = parser.parse_args()
    
    if args.label == 'cataract1k':
        main_cataract1k(args.input, args.output, use_original_fps=args.original_fps)
    elif args.label == 'cataract101':
        main_cataract101(args.input, args.output, use_original_fps=args.original_fps)
    elif args.label == 'CATARACTs':
        main_cataracts_train(args.input, args.output, use_original_fps=args.original_fps)
    else:
        main_others(args.input, args.output, use_original_fps=args.original_fps)