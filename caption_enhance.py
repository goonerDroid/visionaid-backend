# The CaptionEnhance class is designed to enhance the output from Azure's Computer Vision API by adding more context and descriptive elements
# to the generated captions. It processes both the main caption and dense captions to make them more verbose and user-friendly.
from random import choice


class CaptionEnhance:
    # List of introductory phrases for variety
    INTRO_PHRASES = [
        "In this picture we can see",
        "The image shows",
        "This photograph captures",
        "The scene displays",
        "This view presents",
        "Looking at this image, we observe",
        "The photograph reveals",
        "In this scene, we can identify",
        "This snapshot features",
        "Visible in this image is",
        "The picture depicts",
        "Within this frame, we notice",
        "This visual captures",
        "The image portrays",
        "From this perspective, we can see"
    ]

    @staticmethod
    def get_random_intro() -> str:
        """Returns a random introductory phrase"""
        return choice(CaptionEnhance.INTRO_PHRASES)

    @staticmethod
    def clean_caption_text(text: str) -> str:
        """Clean caption text by detecting and removing repetitive patterns"""
        if not text:
            return text

        words = text.split()
        if len(words) <= 3:  # Too short to have repetition
            return text

        # Look for repeating patterns
        clean_words = []
        i = 0
        seen_patterns = set()

        while i < len(words):
            current_pattern = []
            pattern_found = False

            # Try different pattern lengths
            for pattern_length in range(2, min(8, len(words) - i)):
                potential_pattern = tuple(words[i:i + pattern_length])

                # Check if this pattern repeats immediately after
                if (i + pattern_length * 2) <= len(words):
                    next_segment = tuple(
                        words[i + pattern_length:i + pattern_length * 2])
                    if potential_pattern == next_segment:
                        pattern_found = True
                        if potential_pattern not in seen_patterns:
                            clean_words.extend(list(potential_pattern))
                            seen_patterns.add(potential_pattern)
                        i += pattern_length * 2
                        break

            if not pattern_found:
                clean_words.append(words[i])
                i += 1

        return " ".join(clean_words)

    @staticmethod
    def enhance_caption(caption: str) -> str:
        """Add dynamic context to the caption"""
        if not caption:
            return caption

        # Clean the caption text first
        cleaned_caption = CaptionEnhance.clean_caption_text(caption)

        # Check if it already has an intro phrase
        lower_caption = cleaned_caption.lower()
        for phrase in CaptionEnhance.INTRO_PHRASES:
            if lower_caption.startswith(phrase.lower()):
                return cleaned_caption

        return f"{CaptionEnhance.get_random_intro()} {cleaned_caption}"

    @staticmethod
    def enhance_dense_captions(dense_captions: list) -> list:
        """Enhance only the top 2 dense captions with highest confidence scores"""
        if not dense_captions:
            return []

        # Sort dense_captions by confidence in descending order
        sorted_captions = sorted(
            dense_captions,
            key=lambda x: x['confidence'] if x['confidence'] is not None else -1,
            reverse=True
        )

        # Create a new list with enhanced captions
        enhanced = []
        for i, caption in enumerate(sorted_captions):
            # Clean the caption text first
            cleaned_text = CaptionEnhance.clean_caption_text(caption['text'])

            if i < 2:  # Only enhance the top 2 captions
                # Check if it already has an intro phrase
                has_intro = any(
                    cleaned_text.lower().startswith(phrase.lower())
                    for phrase in CaptionEnhance.INTRO_PHRASES
                )

                if not has_intro:
                    cleaned_text = f"{CaptionEnhance.get_random_intro()} {
                        cleaned_text}"

            enhanced.append({
                "text": cleaned_text,
                "confidence": caption['confidence']
            })

        return enhanced

    def enhance_response(self, response: dict) -> dict:
        """Enhance the complete vision API response"""
        enhanced_response = response.copy()

        if 'caption' in enhanced_response:
            enhanced_response['caption'] = self.enhance_caption(
                enhanced_response['caption'])

        if 'dense_captions' in enhanced_response:
            enhanced_response['dense_captions'] = self.enhance_dense_captions(
                enhanced_response['dense_captions'])

        return enhanced_response
