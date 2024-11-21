from typing import Dict, Any
import ast
import json
import re


def extract_json_from_string(s: str) -> Dict[str, Any]:
    """
    Extracts the first JSON-like object from the given string.

    Parameters:
        s (str): The string to be parsed.

    Returns:
        Dict[str, Any]: The parsed JSON object or an empty dictionary if not found.
    """
    start_idx = s.find('{')
    end_idx = s.rfind('}')

    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"String does not contain valid brackets: {s}")

    s = s[start_idx:end_idx + 1]
    s = s.strip()

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        pass

    # Improved regex pattern
    pattern = r"['\"]?(\w+)['\"]?\s*:\s*['\"]?([^,'\"{}]+)['\"]?"
    matches = re.findall(pattern, s)
    result = {match[0]: match[1] for match in matches}

    # If we couldn't extract any key-value pairs, raise ValueError
    if not result:
        return {}

    return result
