#!/usr/bin/env python
"""
Prediction Script

Run inference on sky images for power prediction.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from sky_power_estimation.inference import Predictor
from sky_power_estimation.utils.image_processor import ImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Predict DC Power from Sky Images")
    
    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Input image(s) or directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (CSV)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Ambient temperature (°C)"
    )
    parser.add_argument(
        "--humidity",
        type=float,
        default=50.0,
        help="Relative humidity (%%)"
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=37.7749,
        help="Location latitude"
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=-122.4194,
        help="Location longitude"
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="Timezone"
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Compute uncertainty estimates"
    )
    parser.add_argument(
        "--cloud-features",
        action="store_true",
        help="Extract and display cloud features"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    return parser.parse_args()


def get_image_files(inputs: list) -> list:
    """Get all image files from inputs."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = []
    
    for input_path in inputs:
        path = Path(input_path)
        
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                image_files.append(path)
        elif path.is_dir():
            for ext in image_extensions:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def main():
    args = parse_args()
    
    # Get image files
    image_files = get_image_files(args.input)
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} image(s)")
    
    # Setup location
    location = {
        "latitude": args.latitude,
        "longitude": args.longitude,
        "timezone": args.timezone
    }
    
    # Create predictor
    print(f"Loading model from {args.model}...")
    
    import torch
    device = torch.device(args.device) if args.device else None
    
    predictor = Predictor(
        model_path=args.model,
        device=device,
        location=location
    )
    
    print("Model loaded successfully!")
    
    # Weather data (simplified - in practice, get from sensors)
    weather = {
        "temperature": args.temperature,
        "humidity": args.humidity,
        "pressure": 1013.0,
        "wind_speed": 5.0,
        "ghi": 500.0,  # Placeholder
        "dni": 400.0,
        "dhi": 100.0
    }
    
    # Process images
    results = []
    image_processor = ImageProcessor()
    
    print("\nProcessing images...")
    print("-" * 60)
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {image_path.name}")
        
        try:
            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))
            
            # Get cloud features
            if args.cloud_features:
                cloud_features = predictor.get_cloud_features(image)
                print(f"  Cloud cover: {cloud_features['cloud_cover']:.1f}%")
                print(f"  Brightness: {cloud_features['brightness']:.3f}")
                print(f"  Blue ratio: {cloud_features['blue_ratio']:.3f}")
            
            # Predict
            timestamp = datetime.now()
            
            if args.uncertainty:
                mean_power, std_power = predictor.predict_with_uncertainty(
                    image=image,
                    weather=weather,
                    timestamp=timestamp,
                    num_samples=20
                )
                print(f"  Predicted power: {mean_power:.2f} ± {std_power:.2f} W")
                
                result = {
                    "file": str(image_path),
                    "power_mean": mean_power,
                    "power_std": std_power,
                    "timestamp": timestamp.isoformat()
                }
            else:
                power = predictor.predict(
                    image=image,
                    weather=weather,
                    timestamp=timestamp
                )
                print(f"  Predicted power: {power:.2f} W")
                
                result = {
                    "file": str(image_path),
                    "power": power,
                    "timestamp": timestamp.isoformat()
                }
            
            if args.cloud_features:
                result.update(cloud_features)
            
            results.append(result)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Save results
    if args.output and results:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        
        if args.uncertainty:
            powers = [r["power_mean"] for r in results]
        else:
            powers = [r["power"] for r in results]
        
        print(f"Images processed: {len(results)}")
        print(f"Mean power: {np.mean(powers):.2f} W")
        print(f"Min power: {np.min(powers):.2f} W")
        print(f"Max power: {np.max(powers):.2f} W")


if __name__ == "__main__":
    main()
