from random import choice


class CaptionEnhance:
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
    def get_random_intro():
        return choice(CaptionEnhance.INTRO_PHRASES)

    @staticmethod
    def clean_caption_text(text):
        if not text:
            return text

        words = text.split()
        if len(words) <= 3:
            return text

        clean_words = []
        i = 0
        seen_patterns = set()

        while i < len(words):
            pattern_found = False

            for pattern_length in range(2, min(8, len(words) - i)):
                potential_pattern = tuple(words[i:i + pattern_length])

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
    def enhance_caption(caption):
        if not caption:
            return caption

        cleaned_caption = CaptionEnhance.clean_caption_text(caption)

        lower_caption = cleaned_caption.lower()
        for phrase in CaptionEnhance.INTRO_PHRASES:
            if lower_caption.startswith(phrase.lower()):
                return cleaned_caption

        return f"{CaptionEnhance.get_random_intro()} {cleaned_caption}"

    @staticmethod
    def enhance_dense_captions(dense_captions):
        if not dense_captions:
            return []

        sorted_captions = sorted(
            dense_captions,
            key=lambda x: x['confidence'] if x['confidence'] is not None else -1,
            reverse=True
        )

        enhanced = []
        for i, caption in enumerate(sorted_captions):
            cleaned_text = CaptionEnhance.clean_caption_text(caption['text'])

            if i < 2:
                has_intro = any(cleaned_text.lower().startswith(
                    phrase.lower()) for phrase in CaptionEnhance.INTRO_PHRASES)
                if not has_intro:
                    cleaned_text = f"{CaptionEnhance.get_random_intro()} {
                        cleaned_text}"

            enhanced.append({
                "text": cleaned_text,
                "confidence": caption['confidence']
            })

        return enhanced

    def enhance_response(self, response):
        enhanced_response = response.copy()

        if 'caption' in enhanced_response:
            enhanced_response['caption'] = self.enhance_caption(
                enhanced_response['caption'])

        if 'dense_captions' in enhanced_response:
            enhanced_response['dense_captions'] = self.enhance_dense_captions(
                enhanced_response['dense_captions'])

        return enhanced_response
