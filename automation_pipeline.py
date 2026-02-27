#!/usr/bin/env python3
"""
Complete automation pipeline for advanced lyric video generation
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Replace the analyzer import
from template_analyzer import ComprehensiveFrameAnalyzer
from adaptive_generator import AdaptiveVideoGenerator

class LyricVideoAutomation:
    def __init__(self, original_video_path):
        self.original_video_path = original_video_path
        self.original_video = Path(original_video_path)
        if not self.original_video.exists():
            raise FileNotFoundError(f"Original video not found: {original_video_path}")
        
        self.template_ready = False
        self.generator = None
    
    def setup_template(self, force_reanalyze=False):
        """Setup: Analyze original video to create comprehensive template"""
        template_file = Path('template_assets/comprehensive_template_data.pkl')
        
        if template_file.exists() and not force_reanalyze:
            print("✅ Comprehensive template already exists.")
            print("   Use --force-reanalyze to re-analyze with latest algorithms.")
            self.template_ready = True
            return
        
        print("\n🎬 STARTING COMPREHENSIVE FRAME-BY-FRAME ANALYSIS")
        print("⚠️  This analyzes EVERY frame - estimated time: 10-60 minutes")
        print(f"📹 Video: {self.original_video_path}")
        
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveFrameAnalyzer(self.original_video_path)
        
        # Run complete analysis
        self.template = analyzer.run_complete_analysis()
        
        self.template_ready = True
        print("\n🎉 COMPREHENSIVE TEMPLATE ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {analyzer.total_frames:,} frames")
        print(f"💾 Template data: ~{analyzer._calculate_data_size():.1f} MB")
        print(f"📁 All data saved to: template_assets/")
    
    def generate_single_video(self, audio_path, output_path=None):
        """Generate a single lyric video"""
        if not self.template_ready:
            self.setup_template()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if output_path is None:
            output_path = f"output/{audio_path.stem}_lyric_video.mp4"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n→ Generating lyric video for: {audio_path.name}")
        
        # Initialize generator if needed
        if self.generator is None:
            self.generator = AdaptiveVideoGenerator()
        
        # Generate with quality matching
        quality_score = self.generator.generate_with_quality_matching(
            str(audio_path),
            str(output_path)
        )
        
        print(f"\n🎉 Video generation complete!")
        print(f"   Output: {output_path}")
        print(f"   Quality Score: {quality_score*100:.2f}%")
        
        return str(output_path), quality_score
    
    def batch_process(self, input_folder, output_folder="output"):
        """Process multiple audio files in batch"""
        if not self.template_ready:
            self.setup_template()
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"⚠ No audio files found in {input_folder}")
            return
        
        print(f"\n→ Found {len(audio_files)} audio files to process")
        print(f"→ Output folder: {output_folder}")
        
        # Initialize generator
        if self.generator is None:
            self.generator = AdaptiveVideoGenerator()
        
        results = []
        successful = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n{'='*70}")
            print(f" PROCESSING {i}/{len(audio_files)}: {audio_file.name}")
            print(f"{'='*70}")
            
            try:
                output_file = output_path / f"{audio_file.stem}_lyric_video.mp4"
                video_path, quality_score = self.generate_single_video(
                    str(audio_file),
                    str(output_file)
                )
                
                results.append({
                    'audio_file': str(audio_file),
                    'video_file': video_path,
                    'quality_score': quality_score,
                    'status': 'success'
                })
                successful += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {audio_file.name}: {str(e)}")
                results.append({
                    'audio_file': str(audio_file),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save batch results
        results_file = output_path / 'batch_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*70}")
        print(" BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successful: {successful}/{len(audio_files)}")
        print(f"📊 Results saved to: {results_file}")
        
        if successful > 0:
            avg_quality = np.mean([r['quality_score'] for r in results if r['status'] == 'success'])
            print(f"📈 Average Quality Score: {avg_quality*100:.2f}%")
    
    def watch_folder(self, watch_folder="input", output_folder="output", check_interval=10):
        """
        Continuously monitor a folder for new audio files and process them automatically
        """
        if not self.template_ready:
            self.setup_template()
        
        watch_path = Path(watch_folder)
        output_path = Path(output_folder)
        
        # Create folders if they don't exist
        watch_path.mkdir(exist_ok=True)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(" FOLDER WATCH MODE ACTIVATED")
        print(f"{'='*70}")
        print(f"📁 Watching: {watch_path.absolute()}")
        print(f"📁 Output: {output_path.absolute()}")
        print(f"⏱  Check interval: {check_interval} seconds")
        print(f"🛑 Press Ctrl+C to stop")
        print(f"{'='*70}")
        
        # Initialize generator
        if self.generator is None:
            self.generator = AdaptiveVideoGenerator()
        
        processed_files = set()
        
        try:
            while True:
                # Scan for new audio files
                audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
                new_files = []
                
                for ext in audio_extensions:
                    for audio_file in watch_path.glob(f'*{ext}'):
                        if str(audio_file) not in processed_files:
                            new_files.append(audio_file)
                    for audio_file in watch_path.glob(f'*{ext.upper()}'):
                        if str(audio_file) not in processed_files:
                            new_files.append(audio_file)
                
                # Process new files
                for audio_file in new_files:
                    print(f"\n🆕 New file detected: {audio_file.name}")
                    
                    try:
                        output_file = output_path / f"{audio_file.stem}_lyric_video.mp4"
                        video_path, quality_score = self.generate_single_video(
                            str(audio_file),
                            str(output_file)
                        )
                        
                        print(f"✅ Processing complete: {audio_file.name}")
                        print(f"   Quality Score: {quality_score*100:.2f}%")
                        
                        # Mark as processed
                        processed_files.add(str(audio_file))
                        
                        # Optionally move processed file to a 'processed' subfolder
                        processed_folder = watch_path / "processed"
                        processed_folder.mkdir(exist_ok=True)
                        
                        new_location = processed_folder / audio_file.name
                        audio_file.rename(new_location)
                        print(f"   Moved to: {new_location}")
                        
                    except Exception as e:
                        print(f"❌ Error processing {audio_file.name}: {str(e)}")
                        
                        # Move failed files to error folder
                        error_folder = watch_path / "error"
                        error_folder.mkdir(exist_ok=True)
                        audio_file.rename(error_folder / audio_file.name)
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Folder watch stopped by user")
            print("👋 Goodbye!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced Lyric Video Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Setup - Analyze your original video (run this first)
    python automation_pipeline.py --original original.mp4 --setup
    
    # Generate single video
    python automation_pipeline.py --original original.mp4 --audio song.mp3 --output my_video.mp4
    
    # Batch process folder of songs
    python automation_pipeline.py --original original.mp4 --batch songs/ --output-folder videos/
    
    # Watch folder for automatic processing (recommended for production)
    python automation_pipeline.py --original original.mp4 --watch input/ --output-folder output/
    
    # Force re-analyze template (if you want to update the template)
    python automation_pipeline.py --original original.mp4 --setup --force-reanalyze
        """
    )
    
    parser.add_argument("--original", required=True, help="Path to original template video")
    parser.add_argument("--setup", action="store_true", help="Setup: analyze template video")
    parser.add_argument("--force-reanalyze", action="store_true", help="Force re-analysis of template")
    parser.add_argument("--audio", help="Audio file for single video generation")
    parser.add_argument("--output", help="Output path for single video")
    parser.add_argument("--batch", help="Input folder for batch processing")
    parser.add_argument("--output-folder", default="output", help="Output folder for generated videos")
    parser.add_argument("--watch", help="Watch folder for automatic processing")
    parser.add_argument("--watch-interval", type=int, default=10, help="Watch interval in seconds")
    
    args = parser.parse_args()
    
    try:
        # Initialize automation system
        automation = LyricVideoAutomation(args.original)
        
        if args.setup:
            # Setup/analyze template
            automation.setup_template(force_reanalyze=args.force_reanalyze)
        
        elif args.audio:
            # Single video generation
            automation.generate_single_video(args.audio, args.output)
        
        elif args.batch:
            # Batch processing
            automation.batch_process(args.batch, args.output_folder)
        
        elif args.watch:
            # Watch folder mode
            automation.watch_folder(args.watch, args.output_folder, args.watch_interval)
        
        else:
            # No action specified, show help
            parser.print_help()
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
