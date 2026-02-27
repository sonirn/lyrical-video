import cv2
import numpy as np
import json
import pickle
import os
from moviepy.editor import *
from pathlib import Path
import whisper
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

class AdaptiveVideoGenerator:
    def __init__(self, template_path='template_assets/template_data.pkl'):
        """Initialize generator with learned template"""
        print("Loading template data...")
        
        with open(template_path, 'rb') as f:
            self.template = pickle.load(f)
        
        self.whisper_model = None
        self.quality_threshold = 0.90
        self.max_iterations = 5
        
        # KEY REMAPPING FOR COMPREHENSIVE ANALYZER
        stats = self.template.get('comprehensive_statistics', {})
        self.video_props = self.template.get('video_properties', {})
        self.text_props = stats.get('text', {})
        self.bg_props = stats.get('background', {})
        self.anim_props = stats.get('animation', {})
        
        print("✓ Template data loaded and remapped successfully")
    
    def load_whisper_model(self):
        """Load Whisper AI for audio transcription"""
        if self.whisper_model is None:
            print("Loading Whisper AI model (this may take a moment)...")
            self.whisper_model = whisper.load_model("medium", device="cuda")
            print("✓ Whisper model loaded")
    
    def generate_with_quality_matching(self, audio_path, output_path):
        """Generate video with iterative quality improvement"""
        print("\n" + "="*70)
        print("ADAPTIVE VIDEO GENERATION WITH QUALITY MATCHING")
        print("="*70)
        
        original_video_path = 'original.mp4'
        if not os.path.exists(original_video_path):
            print("⚠ Original video not found for comparison. Generating without quality matching...")
            return self._generate_video_from_template(audio_path, output_path)
        
        best_score = 0
        best_video = None
        current_params = self._get_initial_parameters()
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- ITERATION {iteration}/{self.max_iterations} ---")
            temp_output = f"temp_iter_{iteration}.mp4"
            self._generate_video_with_params(audio_path, temp_output, current_params)
            
            print("Comparing with original video...")
            similarity_score = self._compare_videos(original_video_path, temp_output)
            print(f"Similarity Score: {similarity_score*100:.2f}%")
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_video = temp_output
            
            if similarity_score >= self.quality_threshold:
                print(f"\n🎉 Quality threshold achieved!")
                break
            
            if iteration < self.max_iterations:
                current_params = self._adjust_parameters(original_video_path, temp_output, current_params, similarity_score)
        
        if best_video and os.path.exists(best_video):
            import shutil
            shutil.copy(best_video, output_path)
            print(f"\n✅ Final video saved: {output_path}")
        return best_score

    def _get_initial_parameters(self):
        """Get initial parameters from the NEW text dictionary structure"""
        # Note: The comprehensive analyzer stores text colors differently
        color_data = self.text_props.get('colors', {})
        return {
            'font_size': self.text_props.get('estimated_font_size', 60),
            'text_color': tuple(color_data.get('text_color', [255, 255, 255])),
            'outline_color': tuple(color_data.get('outline_color', [0, 0, 0])),
            'position_x': self.text_props.get('position', {}).get('x_percent', 0.5),
            'position_y': self.text_props.get('position', {}).get('y_percent', 0.75),
            'stroke_width': 3,
            'fade_duration': 0.3,
            'brightness_adjustment': 0,
            'contrast_adjustment': 1.0
        }

    def _generate_video_with_params(self, audio_path, output_path, params):
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        self.load_whisper_model()
        lyrics_data = self._transcribe_with_timing(audio_path)
        
        bg_file = 'template_assets/background_reference.png'
        if os.path.exists(bg_file):
            background = ImageClip(bg_file, duration=duration)
        else:
            background = ColorClip(size=(self.video_props.get('width', 720), self.video_props.get('height', 1280)), color=(0,0,0), duration=duration)
        
        background = background.resize((self.video_props.get('width', 720), self.video_props.get('height', 1280)))
        
        text_clips = []
        for lyric in lyrics_data:
            if lyric['text'].strip():
                t_clip = TextClip(
                    lyric['text'],
                    fontsize=int(params['font_size']),
                    font='Arial-Bold',
                    color=f"rgb{params['text_color']}",
                    stroke_color=f"rgb{params['outline_color']}",
                    stroke_width=params['stroke_width'],
                    method='caption',
                    align='center',
                    size=(self.video_props.get('width', 720) * 0.9, None)
                ).set_position((params['position_x'], params['position_y']), relative=True).set_start(lyric['start']).set_duration(lyric['duration'])
                
                # Apply learned animation
                anim = self.anim_props.get('type', 'fade').lower()
                t_clip = t_clip.fadein(0.3).fadeout(0.3)
                text_clips.append(t_clip)
        
        final_video = CompositeVideoClip([background] + text_clips).set_audio(audio_clip).set_duration(duration)
        
        # Use CPU encoder if NVENC fails, but keep NVENC as primary
        try:
            final_video.write_videofile(output_path, fps=self.video_props.get('fps', 30), codec="h264_nvenc", audio_codec="aac", threads=8, logger=None)
        except:
            final_video.write_videofile(output_path, fps=self.video_props.get('fps', 30), codec="libx264", audio_codec="aac", threads=8, logger=None)
        
        audio_clip.close()
        background.close()
        final_video.close()

    def _transcribe_with_timing(self, audio_path):
        result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
        return [{'start': s["start"], 'end': s["end"], 'duration': s["end"] - s["start"], 'text': s["text"].strip()} for s in result["segments"]]

    def _compare_videos(self, reference_path, generated_path):
        ref_cap, gen_cap = cv2.VideoCapture(reference_path), cv2.VideoCapture(generated_path)
        ssim_scores = []
        for _ in range(10): # Sample 10 frames
            ret1, f1 = ref_cap.read()
            ret2, f2 = gen_cap.read()
            if not ret1 or not ret2: break
            f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
            s = ssim(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY), data_range=255)
            ssim_scores.append(s)
        ref_cap.release(); gen_cap.release()
        return np.mean(ssim_scores) if ssim_scores else 0

    def _adjust_parameters(self, ref, gen, params, score):
        new_params = params.copy()
        new_params['font_size'] *= 1.02 if score < 0.8 else 1.0
        return new_params

    def _generate_video_from_template(self, audio_path, output_path):
        self._generate_video_with_params(audio_path, output_path, self._get_initial_parameters())
        return 0.0
