"""Animate still images with Ken Burns effects or Stable Video Diffusion."""

import logging
import subprocess
from pathlib import Path

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)

# Ken Burns motion variants cycle through scenes
_MOTIONS = [
    "zoompan=z='min(zoom+0.002,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",  # zoom in centre
    "zoompan=z='if(lte(zoom,1.0),1.3,max(1.001,zoom-0.002))':x='0':y='0'",          # zoom out TL
    "zoompan=z='1.2':x='iw/4-(iw/zoom/2)':y='0'",                                   # pan right
    "zoompan=z='1.2':x='3*iw/4-(iw/zoom/2)':y='ih/4'",                              # pan left-down
    "zoompan=z='min(zoom+0.001,1.2)':x='0':y='ih/2-(ih/zoom/2)'",                   # slow zoom L
]


class VideoAnimator:
    def __init__(self, use_svd: bool = config.SVD_ENABLED):
        self.use_svd = use_svd
        self._svd_pipe = None

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("animation")
    def animate_image(
        self,
        image_path: Path,
        output_path: Path,
        duration: float = config.SCENE_DURATION,
        motion_index: int = 0,
    ) -> Path:
        """Animate a single still image; returns path to an MP4 clip."""
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_svd:
            return self._animate_svd(image_path, output_path)
        return self._animate_ken_burns(image_path, output_path, duration, motion_index)

    def animate_batch(
        self,
        image_paths: list[Path],
        out_dir: Path,
        durations: list[float] | None = None,
    ) -> list[Path]:
        """Animate a list of images; returns ordered list of clip paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        durations = durations or [config.SCENE_DURATION] * len(image_paths)
        clips = []
        for i, (img, dur) in enumerate(zip(image_paths, durations)):
            out = out_dir / f"clip_{i:03d}.mp4"
            self.animate_image(img, out, duration=dur, motion_index=i % len(_MOTIONS))
            clips.append(out)
        return clips

    # ── Ken Burns (FFmpeg) ────────────────────────────────────────────────────

    def _animate_ken_burns(
        self,
        image_path: Path,
        output_path: Path,
        duration: float,
        motion_index: int,
    ) -> Path:
        fps = config.VIDEO_FPS
        frames = int(duration * fps)
        motion = _MOTIONS[motion_index % len(_MOTIONS)]

        # zoompan operates on input frames duplicated to match duration
        vf = (
            f"scale={config.IMAGE_WIDTH}:{config.IMAGE_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={config.IMAGE_WIDTH}:{config.IMAGE_HEIGHT},"
            f"{motion}:d={frames}:s={config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}:fps={fps},"
            "format=yuv420p"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image_path),
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an",
            str(output_path),
        ]
        logger.debug("FFmpeg animate: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.error("FFmpeg error: %s", result.stderr.decode())
            raise RuntimeError(f"FFmpeg animation failed for {image_path.name}")
        return output_path

    # ── Stable Video Diffusion ────────────────────────────────────────────────

    def _animate_svd(self, image_path: Path, output_path: Path) -> Path:
        pipe = self._load_svd()
        from PIL import Image
        import torch, imageio

        image = Image.open(image_path).convert("RGB").resize(
            (1024, 576), Image.LANCZOS          # SVD native resolution
        )
        frames_tensor = pipe(
            image,
            num_frames=config.SVD_FRAMES,
            num_inference_steps=25,
        ).frames[0]

        frames_np = [(f * 255).clip(0, 255).to(torch.uint8).numpy() for f in frames_tensor]
        writer = imageio.get_writer(str(output_path), fps=config.SVD_FPS, codec="libx264")
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        return output_path

    def _load_svd(self):
        if self._svd_pipe is None:
            import torch
            from diffusers import StableVideoDiffusionPipeline

            logger.info("Loading Stable Video Diffusion …")
            self._svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                config.SVD_MODEL,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            self._svd_pipe.enable_model_cpu_offload()
        return self._svd_pipe
