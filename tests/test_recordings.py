"""The Gitleaks exception for the recordings is only safe while a long hex run in them is a key."""

import json
import re
from pathlib import Path

import pytest

# A digits-only run (a score's decimals, say) has too little entropy for Gitleaks to flag, so a
# run counts only when it holds a letter from a to f.
HEX_RUN = re.compile(r"(?<![0-9A-Za-z.])(?=[0-9]*[a-f])[0-9a-f]{10,}(?![0-9A-Za-z])")
RECORDINGS = Path(__file__).parent.parent / "results" / "qasper"


@pytest.mark.parametrize("name", ["calls.jsonl", "runs.jsonl"])
def test_the_only_hex_run_of_10_or_more_in_a_recording_line_is_its_key(name):
    path = RECORDINGS / name
    if not path.exists():
        pytest.skip("no recording committed yet")
    for n, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        record = json.loads(line)
        found = HEX_RUN.findall(line)
        key = record["key"]
        if re.fullmatch(r"[0-9a-f]{64}", key):  # a request hash, or a search hash
            expected = [key]
        elif re.fullmatch(r"[0-9a-f]{40}\|(single pass|loop)", key):  # a stop: question id and system
            expected = [key.split("|")[0]]
        else:
            expected = []
        assert found == expected, f"{name} line {n}"
