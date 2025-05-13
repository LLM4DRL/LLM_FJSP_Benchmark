# Utility functions for calculating token metrics

def count_tokens(text):
    """Count tokens in a text using simple whitespace splitting."""
    return len(text.split())


def calculate_tokens_per_sec(input_text, output_text, elapsed_time):
    """Calculate tokens per second based on input and output text and elapsed time."""
    total_tokens = count_tokens(input_text) + count_tokens(output_text)
    return total_tokens / elapsed_time if elapsed_time > 0 else 0.0 