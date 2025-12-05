# scripts/splitdataset.py
import json
import argparse
from pathlib import Path

def split_dataset(json_path, test_skip=8, mode='auto'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    n_frames = len(frames)
    
    if mode == 'auto' and n_frames < 100:
        print(f"Using optimized split for small datasets\n")
        
        test_indices = set(range(0, n_frames, n_frames // 2))  # 2 images
        val_indices = set(range(1, n_frames, n_frames // 2))   # 2 images
        
        train_frames = []
        val_frames = []
        test_frames = []
        
        for i, frame in enumerate(frames):
            if i in test_indices:
                test_frames.append(frame)
            elif i in val_indices:
                val_frames.append(frame)
            else:
                train_frames.append(frame)
    else:
        train_frames = []
        val_frames = []
        test_frames = []
        
        for i, frame in enumerate(frames):
            if i % test_skip == 0:
                test_frames.append(frame)
            elif i % test_skip == test_skip // 2:
                val_frames.append(frame)
            else:
                train_frames.append(frame)
    
    print(f"Total frames: {n_frames}")
    print(f"Train: {len(train_frames)} ({len(train_frames)/n_frames*100:.1f}%)")
    print(f"Val: {len(val_frames)} ({len(val_frames)/n_frames*100:.1f}%)")
    print(f"Test: {len(test_frames)} ({len(test_frames)/n_frames*100:.1f}%)")
    
    base_path = Path(json_path).parent
    
    train_data = data.copy()
    train_data['frames'] = train_frames
    with open(base_path / 'transforms_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Created: {base_path / 'transforms_train.json'}")
    
    if val_frames:
        val_data = data.copy()
        val_data['frames'] = val_frames
        with open(base_path / 'transforms_val.json', 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Created: {base_path / 'transforms_val.json'}")
    
    test_data = data.copy()
    test_data['frames'] = test_frames
    with open(base_path / 'transforms_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Created: {base_path / 'transforms_test.json'}")

def main():
    parser = argparse.ArgumentParser(description='Split NeRF dataset')
    parser.add_argument('--json_path', type=str, required=True, 
                       help='Path to transforms.json')
    parser.add_argument('--test_skip', type=int, default=8, 
                       help='Take 1 out of every N images for test set')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['auto', 'standard'],
                       help='auto: optimize for small datasets, standard: use test_skip')
    
    args = parser.parse_args()
    split_dataset(args.json_path, args.test_skip, args.mode)

if __name__ == '__main__':
    main()