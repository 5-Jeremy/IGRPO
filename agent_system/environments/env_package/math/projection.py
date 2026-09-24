"""Keep one complete math action and reject search or mixed tool calls."""

import re


ACTION = re.compile(r"<(python|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
OPEN = re.compile(r"<(python|answer|search)>", re.IGNORECASE)


def math_projection(actions):
    projected, valid = [], []
    for action in actions:
        matches = list(ACTION.finditer(action))
        tags = OPEN.findall(action)
        ok = len(matches) == 1 and len(tags) == 1 and tags[0].lower() != "search"
        if ok and matches[0].group(1).lower() == "python" and not matches[0].group(2).strip():
            ok = False
        projected.append(f"<{matches[0].group(1).lower()}>{matches[0].group(2).strip()}</{matches[0].group(1).lower()}>" if ok else "")
        valid.append(int(ok))
    return projected, valid
