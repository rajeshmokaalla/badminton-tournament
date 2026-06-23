"""Image generation using SDXL or FLUX via diffusers."""

import logging
from pathlib import Path

import config
from utils.helpers import timed

logger = logging.getLogger(__name__)

# Common negative prompt applied to all generations
_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, text overlay, "
    "nsfw, violence, ugly, deformed, extra limbs"
)


class ImageGenerator:
    def __init__(
        self,
        backend: str = config.IMAGE_BACKEND,
        width: int = config.IMAGE_WIDTH,
        height: int = config.IMAGE_HEIGHT,
    ):
        self.backend = backend.lower()
        self.width = width
        self.height = height
        self._pipe = None           # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────────

    @timed("image")
    def generate(self, prompt: str, output_path: Path, seed: int | None = None) -> Path:
        """Generate a single image from *prompt* and save to *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Generating image: %r", self.backend.upper(), prompt[:80])
        try:
            pipe = self._load_pipeline()
            kwargs = dict(
                prompt=prompt,
                negative_prompt=_NEGATIVE_PROMPT,
                width=self.width,
                height=self.height,
                num_inference_steps=self._steps(),
                guidance_scale=self._guidance(),
            )
            if seed is not None:
                import torch
                kwargs["generator"] = torch.Generator(device=self._device()).manual_seed(seed)

            result = pipe(**kwargs)
            image = result.images[0]
            image.save(str(output_path))

        except Exception as exc:
            logger.warning("Image generation failed (%s); using placeholder", exc)
            self._write_placeholder(output_path)

        logger.info("Image saved: %s", output_path)
        return output_path

    def generate_batch(
        self,
        prompts: list[str],
        out_dir: Path,
        base_name: str = "scene",
    ) -> list[Path]:
        """Generate one image per prompt; returns ordered list of paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, prompt in enumerate(prompts):
            p = out_dir / f"{base_name}_{i:03d}.png"
            self.generate(prompt, p, seed=i)
            paths.append(p)
        return paths

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import (
            StableDiffusionXLPipeline,
            FluxPipeline,
            DPMSolverMultistepScheduler,
        )

        device = self._device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        if self.backend == "flux":
            logger.info("Loading FLUX pipeline …")
            self._pipe = FluxPipeline.from_pretrained(
                config.FLUX_MODEL, torch_dtype=dtype
            ).to(device)
        else:
            logger.info("Loading SDXL pipeline …")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                config.SDXL_MODEL,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if dtype == torch.float16 else None,
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            if device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(device)
            self._pipe = pipe

        return self._pipe

    def _steps(self) -> int:
        return 4 if self.backend == "flux" else 30

    def _guidance(self) -> float:
        return 0.0 if self.backend == "flux" else 7.5

    @staticmethod
    def _device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _write_placeholder(path: Path) -> None:
        """Create a solid-colour PNG when generation is unavailable."""
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), color=(30, 30, 50))
            draw = ImageDraw.Draw(img)
            draw.text((config.IMAGE_WIDTH // 2, config.IMAGE_HEIGHT // 2),
                      "AI Video\nPipeline", fill=(200, 200, 200), anchor="mm")
            img.save(str(path))
        except ImportError:
            # Last resort: write a minimal 1×1 PNG
            import struct, zlib
            def _png_chunk(tag, data):
                c = zlib.crc32(tag + data) & 0xFFFFFFFF
                return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)
            raw = (
                b"\x89PNG\r\n\x1a\n"
                + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                + _png_chunk(b"IDAT", zlib.compress(b"\x00\x1e\x1e\x32"))
                + _png_chunk(b"IEND", b"")
            )
            path.write_bytes(raw)
