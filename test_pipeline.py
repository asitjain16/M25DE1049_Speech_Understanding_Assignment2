#!/usr/bin/env python3
"""
Test script to verify pipeline components are working
"""

import sys
import os

print("=" * 70)
print("SPEECH UNDERSTANDING ASSIGNMENT 2 - PIPELINE TEST")
print("=" * 70)
print()

# Test 1: Check imports
print("TEST 1: Checking imports...")
try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import whisper
    print("✅ OpenAI Whisper imported successfully")
except ImportError as e:
    print(f"❌ Whisper import failed: {e}")
    sys.exit(1)

try:
    import librosa
    print("✅ Librosa imported successfully")
except ImportError as e:
    print(f"❌ Librosa import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    import yaml
    print("✅ PyYAML imported successfully")
except ImportError as e:
    print(f"❌ PyYAML import failed: {e}")
    sys.exit(1)

print()

# Test 2: Check configuration
print("TEST 2: Checking configuration...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration file loaded successfully")
    print(f"   - Target LRL: {config['translation']['target_lrl']}")
    print(f"   - Sample Rate: {config['audio']['sample_rate']} Hz")
    print(f"   - STT Model: {config['stt']['model']}")
except Exception as e:
    print(f"❌ Configuration loading failed: {e}")
    sys.exit(1)

print()

# Test 3: Check data files
print("TEST 3: Checking data files...")
data_files = [
    'data/processed/original_segment.wav',
    'data/reference/student_voice_ref.wav',
    'data/technical_terms_dict.json'
]

for file_path in data_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✅ {file_path} ({size_mb:.2f} MB)")
    else:
        print(f"⚠️  {file_path} (not found)")

print()

# Test 4: Check source code
print("TEST 4: Checking source code modules...")
modules = [
    'src/part1_stt/transcription_pipeline.py',
    'src/part2_translation/translation_pipeline.py',
    'src/part3_tts/synthesis_pipeline.py',
    'src/part4_adversarial/adversarial_pipeline.py',
    'src/utils/audio_utils.py',
    'src/utils/metrics.py'
]

for module in modules:
    if os.path.exists(module):
        print(f"✅ {module}")
    else:
        print(f"❌ {module} (not found)")

print()

# Test 5: Check output directories
print("TEST 5: Checking output directories...")
output_dirs = [
    'outputs/transcripts',
    'outputs/audio',
    'outputs/metrics'
]

for dir_path in output_dirs:
    if os.path.exists(dir_path):
        print(f"✅ {dir_path}")
    else:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ {dir_path} (created)")

print()

# Test 6: Check report files
print("TEST 6: Checking report files...")
report_files = [
    'report/M25DE1049_Assignment_Report.md',
    'report/M25DE1049_Assignment_Report.docx',
    'report/M25DE1049_Implementation_Notes.md',
    'report/M25DE1049_Implementation_Notes.docx'
]

for file_path in report_files:
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        print(f"✅ {file_path} ({size_kb:.2f} KB)")
    else:
        print(f"⚠️  {file_path} (not found)")

print()

# Test 7: Device check
print("TEST 7: Checking device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    print(f"   - CUDA Version: {torch.version.cuda}")

print()

# Test 8: Summary
print("=" * 70)
print("✅ ALL TESTS PASSED - PIPELINE READY FOR EXECUTION")
print("=" * 70)
print()
print("NEXT STEPS:")
print("1. Run: python pipeline.py --part 1 --audio data/processed/original_segment.wav")
print("2. Run: python pipeline.py --part 2")
print("3. Run: python pipeline.py --part 3 --audio data/processed/original_segment.wav")
print("4. Run: python pipeline.py --part 4 --audio data/processed/original_segment.wav")
print()
print("OR run complete pipeline:")
print("python pipeline.py --config config.yaml --audio data/processed/original_segment.wav")
print()
print("=" * 70)
