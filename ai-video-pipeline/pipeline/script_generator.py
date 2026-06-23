"""Script generation via Ollama (Llama 3, Qwen 2.5, etc.)."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


@dataclass
class ScriptScene:
    index: int
    narration: str          # spoken text for this scene
    image_prompt: str       # Stable Diffusion / FLUX prompt
    duration: float = config.SCENE_DURATION


@dataclass
class VideoScript:
    title: str
    description: str
    hashtags: list[str]
    scenes: list[ScriptScene]
    music_prompt: str       # MusicGen prompt
    raw_topic: str = ""

    def full_narration(self) -> str:
        return " ".join(s.narration for s in self.scenes)


SYSTEM_PROMPT = """\
You are a creative director specialising in viral short-form video (YouTube Shorts / Instagram Reels).
Given a topic, produce a complete video script in strict JSON with the following schema:

{
  "title": "<catchy title ≤ 60 chars>",
  "description": "<1-2 sentence description>",
  "hashtags": ["<tag1>", "<tag2>", ...],   // 5-10 tags without #
  "music_prompt": "<MusicGen text prompt for background music>",
  "scenes": [
    {
      "index": 0,
      "narration": "<15-25 words of spoken narration>",
      "image_prompt": "<detailed Stable Diffusion prompt, photorealistic, 9:16>"
    }
    // 5-8 scenes total
  ]
}

Rules:
- Keep total narration under 90 seconds when spoken aloud (~150 wpm).
- Image prompts must be vivid, safe-for-work, and vertical-friendly (portrait orientation).
- Music prompt: 10-20 words describing mood, genre, instruments.
- Return ONLY valid JSON, no markdown fences.
"""


class ScriptGenerator:
    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, topic: str, style: str = "engaging and informative") -> VideoScript:
        """Generate a complete VideoScript for *topic*."""
        logger.info("Generating script for topic: %s", topic)
        user_prompt = f"Topic: {topic}\nStyle: {style}\n\nGenerate the JSON script now."
        raw = self._call_ollama(user_prompt)
        script = self._parse_response(raw, topic)
        logger.info("Script ready: %d scenes, title=%r", len(script.scenes), script.title)
        return script

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_ollama(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.85, "top_p": 0.9},
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable; using fallback script")
            return self._fallback_script(user_prompt)

    def _parse_response(self, raw: str, topic: str) -> VideoScript:
        # Strip accidental markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed; attempting extraction")
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                logger.error("Could not parse JSON; using fallback")
                data = json.loads(self._fallback_script(topic))

        scenes = [
            ScriptScene(
                index=s.get("index", i),
                narration=s.get("narration", ""),
                image_prompt=s.get("image_prompt", topic),
                duration=config.SCENE_DURATION,
            )
            for i, s in enumerate(data.get("scenes", []))
        ]
        return VideoScript(
            title=data.get("title", topic[:60]),
            description=data.get("description", ""),
            hashtags=data.get("hashtags", []),
            scenes=scenes,
            music_prompt=data.get("music_prompt", "uplifting cinematic background music"),
            raw_topic=topic,
        )

    def _fallback_script(self, topic: str) -> str:
        """Minimal fallback if Ollama is unavailable."""
        data = {
            "title": f"Amazing facts about {topic}",
            "description": f"Discover incredible things about {topic} in 60 seconds.",
            "hashtags": ["shorts", "facts", "trending", "viral", "learn"],
            "music_prompt": "upbeat background music, modern, energetic",
            "scenes": [
                {
                    "index": i,
                    "narration": f"Scene {i + 1}: Discover something amazing about {topic}.",
                    "image_prompt": (
                        f"Photorealistic portrait image about {topic}, "
                        "vibrant colors, 9:16 aspect ratio, high quality"
                    ),
                }
                for i in range(5)
            ],
        }
        return json.dumps(data)
