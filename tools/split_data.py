import argparse
import os
import glob
import shutil
import csv
import numpy as np
from collections import defaultdict


def get_phase_distribution(video_path):
    """
    Analyze the phase distribution for a given video path.
    
    Args:
        video_path: Path to the video folder containing CSV files with phase labels
        
    Returns:
        A dictionary mapping phase IDs to counts and the total frame count
    """
    phase_counts = defaultdict(int)
    csv_files = glob.glob(os.path.join(video_path, '*.csv'))
    
    # If no CSV files found, return empty distribution
    if not csv_files:
        return phase_counts, 0
    
    # Get the first CSV file (assuming there's typically one per video)
    csv_file = csv_files[0]
    total_frames = 0
    
    try:
        # Read phase labels from the CSV file (first column)
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)  # Skip header if present
            for row in csv_reader:
                if row:  # Make sure row is not empty
                    phase_id = int(float(row[0]))  # First column contains the phase label
                    phase_counts[phase_id] += 1
                    total_frames += 1
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    
    return phase_counts, total_frames


def analyze_dataset(videos):
    """
    Analyze the dataset to gather statistics about phase distribution.
    
    Args:
        videos: List of video paths
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_videos': len(videos),
        'total_frames': 0,
        'phase_counts': defaultdict(int),
        'videos_by_phase': defaultdict(list),
        'frames_per_video': {},
        'phases_per_video': {}
    }
    
    for video_path in videos:
        video_id = os.path.basename(video_path)
        phase_counts, total_frames = get_phase_distribution(video_path)
        
        if total_frames == 0:
            print(f"Warning: No frames found in {video_id}")
            continue
            
        # Record which phases appear in this video
        stats['frames_per_video'][video_path] = total_frames
        stats['phases_per_video'][video_path] = dict(phase_counts)
            
        # Update statistics
        stats['total_frames'] += total_frames
        for phase, count in phase_counts.items():
            stats['phase_counts'][phase] += count
            # Add video to the list of videos containing this phase
            if count > 0:
                stats['videos_by_phase'][phase].append((video_path, count))
    
    return stats


def print_dataset_stats(stats):
    """Print comprehensive dataset statistics."""
    print("\n===== DATASET SUMMARY =====")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Total frames: {stats['total_frames']}")
    
    if stats['total_videos'] > 0:
        print(f"Average frames per video: {stats['total_frames'] / stats['total_videos']:.1f}")
    
    # Sort phases by frame count (descending)
    sorted_phases = sorted(stats['phase_counts'].items(), 
                           key=lambda x: x[1], reverse=True)
    
    print(f"Number of unique phases: {len(sorted_phases)}")
    
    print("\n===== PHASE DISTRIBUTION =====")
    for phase, count in sorted_phases:
        percentage = (count / stats['total_frames']) * 100
        videos_with_phase = len(stats['videos_by_phase'][phase])
        print(f"Phase {phase}: {count} frames ({percentage:.2f}%) in {videos_with_phase} videos")


def balanced_frame_based_split(videos, stats, val_ratio=0.2, test_ratio=0.1, min_frames_per_phase=100, seed=42):
    """
    Split videos into training, validation, and test sets with a focus on balancing
    the representation of all phases in each split.
    
    Args:
        videos: List of video paths
        stats: Dataset statistics from analyze_dataset()
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        min_frames_per_phase: Target minimum number of frames for each phase in each split
        seed: Random seed for reproducibility
        
    Returns:
        train_videos, val_videos, test_videos: Lists of video paths for each set
    """
    np.random.seed(seed)
    
    all_phases = set(stats['phase_counts'].keys())
    
    available_videos = videos.copy()
    
    train_videos = []
    val_videos = []
    test_videos = []
    
    train_frames_per_phase = defaultdict(int)
    val_frames_per_phase = defaultdict(int)
    test_frames_per_phase = defaultdict(int)
    
    sorted_phases = sorted(all_phases, key=lambda p: stats['phase_counts'][p])
    
    for phase in sorted_phases:
        videos_with_phase = [(v, c) for v, c in stats['videos_by_phase'][phase] if v in available_videos]
        
        if not videos_with_phase:
            continue
            
        videos_with_phase.sort(key=lambda x: x[1], reverse=True)
        
        if len(videos_with_phase) >= 3:
            np.random.shuffle(videos_with_phase)
            
            test_count = max(1, int(len(videos_with_phase) * test_ratio))
            val_count = max(1, int(len(videos_with_phase) * val_ratio))
            train_count = len(videos_with_phase) - val_count - test_count
            
            if train_count < 1:
                train_count = 1
                val_count = max(1, len(videos_with_phase) - train_count - test_count)
            
            test_count = min(test_count, len(videos_with_phase))
            val_count = min(val_count, len(videos_with_phase) - test_count)
            train_count = len(videos_with_phase) - val_count - test_count
            
            for i, (video_path, count) in enumerate(videos_with_phase):
                if video_path not in available_videos:
                    continue
                    
                if i < test_count:
                    test_videos.append(video_path)
                    available_videos.remove(video_path)
                    for p, c in stats['phases_per_video'][video_path].items():
                        test_frames_per_phase[p] += c

                elif i < test_count + val_count:
                    val_videos.append(video_path)
                    available_videos.remove(video_path)
                    for p, c in stats['phases_per_video'][video_path].items():
                        val_frames_per_phase[p] += c

                else:
                    train_videos.append(video_path)
                    available_videos.remove(video_path)
                    for p, c in stats['phases_per_video'][video_path].items():
                        train_frames_per_phase[p] += c
        elif len(videos_with_phase) == 2:
            for i, (video_path, count) in enumerate(videos_with_phase):
                if video_path not in available_videos:
                    continue
                    
                if i == 0:
                    train_videos.append(video_path)
                    available_videos.remove(video_path)  # Mark as used
                    for p, c in stats['phases_per_video'][video_path].items():
                        train_frames_per_phase[p] += c
                else:
                    val_videos.append(video_path)
                    available_videos.remove(video_path)  # Mark as used
                    for p, c in stats['phases_per_video'][video_path].items():
                        val_frames_per_phase[p] += c
        elif len(videos_with_phase) == 1:
            # If only 1 video has this phase, put it in train
            video_path, count = videos_with_phase[0]
            if video_path in available_videos:
                train_videos.append(video_path)
                available_videos.remove(video_path)  # Mark as used
                for p, c in stats['phases_per_video'][video_path].items():
                    train_frames_per_phase[p] += c
    
    # Assign remaining videos to maintain overall split ratios
    if available_videos:
        # Shuffle for randomness
        np.random.shuffle(available_videos)
        
        # Calculate target counts
        total_remaining = len(available_videos)
        target_test = int(total_remaining * test_ratio)
        target_val = int(total_remaining * val_ratio)
        target_train = total_remaining - target_test - target_val
        
        # Assign remaining videos
        for i, video_path in enumerate(available_videos):
            if i < target_test:
                test_videos.append(video_path)
                for p, c in stats['phases_per_video'][video_path].items():
                    test_frames_per_phase[p] += c
            elif i < target_test + target_val:
                val_videos.append(video_path)
                for p, c in stats['phases_per_video'][video_path].items():
                    val_frames_per_phase[p] += c
            else:
                train_videos.append(video_path)
                for p, c in stats['phases_per_video'][video_path].items():
                    train_frames_per_phase[p] += c
    
    # Check for any accidental overlaps
    train_set = set(train_videos)
    val_set = set(val_videos)
    test_set = set(test_videos)
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    # Verify no video is missed
    all_assigned = set(train_videos + val_videos + test_videos)
    if len(all_assigned) != len(videos):
        print(f"WARNING: {len(videos) - len(all_assigned)} videos were not assigned to any split!")
    
    return train_videos, val_videos, test_videos


def analyze_split_distribution(train_videos, val_videos, test_videos, stats):
    """
    Analyze the phase distribution in each split to verify balance.
    
    Args:
        train_videos, val_videos, test_videos: Lists of video paths for each split
        stats: Dataset statistics from analyze_dataset()
    """
    splits = {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }
    
    all_phases = set(stats['phase_counts'].keys())
    split_stats = {}
    
    # First pass: collect per-split statistics
    for split_name, videos in splits.items():
        split_stats[split_name] = {
            'videos': len(videos),
            'frames': 0,
            'phase_counts': defaultdict(int)
        }
        
        for video_path in videos:
            total_frames = stats['frames_per_video'].get(video_path, 0)
            phase_counts = stats['phases_per_video'].get(video_path, {})
            
            split_stats[split_name]['frames'] += total_frames
            for phase, count in phase_counts.items():
                split_stats[split_name]['phase_counts'][phase] += count
    
    # Print statistics for each split
    for split_name, split_stat in split_stats.items():
        print(f"\n===== {split_name.upper()} SPLIT =====")
        print(f"Videos: {split_stat['videos']}")
        print(f"Frames: {split_stat['frames']}")
        
        if split_stat['frames'] > 0:
            print("\nPhase distribution:")
            # Sort phases by count
            sorted_phases = sorted(split_stat['phase_counts'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            for phase, count in sorted_phases:
                percentage = (count / split_stat['frames']) * 100
                print(f"  Phase {phase}: {count} frames ({percentage:.2f}%)")
    
    # Calculate overall phase distribution across all splits
    total_in_splits = sum(s['frames'] for s in split_stats.values())
    
    # For each phase, calculate percent in each split
    print("\n===== PHASE REPRESENTATION ACROSS SPLITS =====")
    for phase in sorted(all_phases):
        print(f"Phase {phase}:")
        total_phase_frames = sum(split_stats[split]['phase_counts'].get(phase, 0) for split in splits)
        
        if total_phase_frames > 0:
            for split_name in splits:
                phase_count = split_stats[split_name]['phase_counts'].get(phase, 0)
                split_percent = (phase_count / total_phase_frames) * 100
                print(f"  {split_name}: {phase_count} frames ({split_percent:.2f}% of phase {phase} frames)")
    
    # Print class distribution in a tabular format for easy comparison
    print("\n===== CLASS DISTRIBUTION SUMMARY TABLE =====")
    
    # Calculate total frames in each split
    total_frames = {split: stats['frames'] for split, stats in split_stats.items()}
    all_total = sum(total_frames.values())
    
    # Header of the table
    header = ["Phase", "Global %"] + [f"{split} %" for split in splits.keys()] + ["Total Frames"]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))
    
    # Row for each phase
    for phase in sorted(all_phases):
        # Global percentage of this phase
        global_percent = (stats['phase_counts'][phase] / stats['total_frames']) * 100
        
        # Percentage in each split
        split_percents = {}
        for split_name in splits:
            phase_count = split_stats[split_name]['phase_counts'].get(phase, 0)
            if split_stats[split_name]['frames'] > 0:
                split_percents[split_name] = (phase_count / split_stats[split_name]['frames']) * 100
            else:
                split_percents[split_name] = 0
        
        # Total frames for this phase
        total_phase_frames = stats['phase_counts'][phase]
        
        # Format the row
        row = [
            f"{phase}",
            f"{global_percent:.2f}%"
        ]
        
        # Add percentage for each split
        for split_name in splits:
            row.append(f"{split_percents[split_name]:.2f}%")
        
        # Add total frames
        row.append(f"{total_phase_frames}")
        
        print(" | ".join(row))


def main():
    parser = argparse.ArgumentParser(description='Split data into train/val/test with balanced phase distribution')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='path to folder containing case_id folders (e.g., data/cataract1k)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='path to output folder (defaults to same as input)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--min_frames_per_phase',
        type=int,
        default=100,
        help='Target minimum number of frames for each phase in each split (default: 100)'
    )
    
    args = parser.parse_args()
    
    if args.out is None:
        args.out = args.input
    
    video_dirs = []
    case_dirs = glob.glob(os.path.join(args.input, 'case_*'))
    if not case_dirs:
        potential_dirs = [d for d in glob.glob(os.path.join(args.input, '*')) 
                         if os.path.isdir(d)]
        
        for d in potential_dirs:
            has_frames = len(glob.glob(os.path.join(d, '*.jpg'))) > 0
            has_csv = len(glob.glob(os.path.join(d, '*.csv'))) > 0
            
            if has_frames and has_csv:
                video_dirs.append(d)
    else:
        video_dirs = case_dirs
    
    if not video_dirs:
        print(f"No video directories found in {args.input}")
        exit(1)
    
    print(f"Found {len(video_dirs)} video directories in {args.input}")
    
    # # Analyze the dataset
    stats = analyze_dataset(video_dirs)
    print_dataset_stats(stats)
    
    # Create output directories if they don't exist
    for phase in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.out, phase), exist_ok=True)
    
    # Perform the balanced frame-based split
    train_videos, val_videos, test_videos = balanced_frame_based_split(
        video_dirs,
        stats,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_frames_per_phase=args.min_frames_per_phase,
        seed=args.seed
    )
    
    print(f"\nInitial split complete: {len(train_videos)} training, {len(val_videos)} validation, "
          f"{len(test_videos)} test videos")
    
    # Analyze the phase distribution in each split
    missing_in_val, missing_in_test = analyze_split_distribution(train_videos, val_videos, test_videos, stats)
    
    # Apply augmentation if requested to ensure all phases are in all splits
    # if args.augment:
    #     train_videos, val_videos, test_videos = ensure_all_phases_in_splits(
    #         train_videos, 
    #         val_videos, 
    #         test_videos, 
    #         stats, 
    #         args.out,
    #         min_examples=args.min_synthetic
    #     )
        
    #     print(f"\nAfter augmentation: {len(train_videos)} training, {len(val_videos)} validation, "
    #           f"{len(test_videos)} test videos")
        
        # # Re-analyze the distribution after augmentation
        # print("\n===== DISTRIBUTION AFTER AUGMENTATION =====")
        # missing_in_val, missing_in_test = analyze_split_distribution(train_videos, val_videos, test_videos, stats)
        
        # if not missing_in_val and not missing_in_test:
        #     print("\nSuccess! All phases are now represented in all splits.")
        # else:
        #     print("\nWarning: Some phases still missing after augmentation.")
        #     if missing_in_val:
        #         print(f"Validation still missing phases: {missing_in_val}")
        #     if missing_in_test:
        #         print(f"Test still missing phases: {missing_in_test}")
    
    moved_success = 0
    
    video_to_split = {}
    for video in train_videos:
        video_to_split[video] = 'train'
    for video in val_videos:
        video_to_split[video] = 'val'
    for video in test_videos:
        video_to_split[video] = 'test'
    
    for video_path, split in video_to_split.items():
        video_id = os.path.basename(video_path)
        dst_path = os.path.join(args.out, split, video_id)
        
        if os.path.normpath(video_path) == os.path.normpath(dst_path):
            continue
            
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)
            
            shutil.move(video_path, dst_path)
            moved_success += 1
        except Exception as e:
            print(f"Failed to move {video_path} to {dst_path}: {e}")
    
    print(f"\nSuccessfully moved {moved_success} video directories out of {len(video_to_split)}")
    
    # Print final directory counts
    train_dir = os.path.join(args.out, 'train')
    val_dir = os.path.join(args.out, 'val')
    test_dir = os.path.join(args.out, 'test')
    
    train_count = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    val_count = len([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    test_count = len([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    print(f"Final directory counts - Train: {train_count}, Val: {val_count}, Test: {test_count}")


if __name__ == "__main__":
    main()