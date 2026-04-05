"""
Quick diagnostic: paste this at the project root and run it ONCE to verify
the IPA parser produces sensible ARPAbet output for the exact tokens
that appeared in your logs.

    python diagnose_ipa.py

Expected output: all green ticks.  If you see failures, paste the output
and I'll fix the parser.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from services.phoneme_recognition_service import _parse_ipa_token, _token_to_arpabet

cases = [
    # (raw_token,          expected_arpabet_contains_at_least)
    ("haɪ",               ["HH", "AY"]),
    ("hjɛ",               ["HH"]),
    ("ɛ",                 ["EH"]),
    ("dɪɪ",               ["D", "IH"]),
    ("bɪs",               ["B", "IH", "S"]),
    ("ən",                ["AH", "N"]),
    ("d",                 ["D"]),
    ("ɑɾ",                ["AA", "R"]),
    ("kɪn",               ["K", "IH", "N"]),
    ("ɪn",                ["IH", "N"]),
    ("dɪ",                ["D", "IH"]),
    ("kɑ",                ["K", "AA"]),
    ("ɑɾɪ",               ["AA", "R", "IH"]),
    ("tɪ",                ["T", "IH"]),
    ("dɪs",               ["D", "IH", "S"]),
    ("k",                 ["K"]),
    ("vu",                ["V", "UW"]),
    ("i",                 ["IY"]),
    ("bɑ",                ["B", "AA"]),
    ("ðə",                ["DH", "AH"]),
    ("ðənɑ",              ["DH", "AH", "N", "AA"]),
    ("s",                 ["S"]),
    ("tʊ",                ["T", "UH"]),
    ("dʊs",               ["D", "UH", "S"]),
    ("pi",                ["P", "IY"]),
    ("nɪŋ",               ["N", "IH", "NG"]),
    ("glɪ",               ["G", "L", "IH"]),
    ("lɪs",               ["L", "IH", "S"]),
    ("bə",                ["B", "AH"]),
    ("kaɪ",               ["K", "AY"]),
    ("ɪnhəs",             ["IH", "N", "HH", "AH", "S"]),
    ("snɑ",               ["S", "N", "AA"]),
    ("ɑnɑlɪ",             ["AA", "N", "AA", "L", "IH"]),
    ("mhoʊ",              ["M", "HH", "OW"]),
    ("oʊsi",              ["OW", "S", "IY"]),
    ("pɝən",              ["P", "ER", "AH", "N"]),
    ("ʤzɪ",               ["JH", "Z", "IH"]),
    ("ɪsəv",              ["IH", "S", "AH", "V"]),
    ("wʊ",                ["W", "UH"]),
    ("pɝnaʊn",            ["P", "ER", "N", "AW", "N"]),
    ("ʤeɪs",              ["JH", "EY", "S"]),
    ("sɪnɪz",             ["S", "IH", "N", "IH", "Z"]),
    ("svɛɾi",             ["S", "V", "EH", "R", "IY"]),
    ("ɑvɛɾ",              ["AA", "V", "EH", "R"]),
    ("aɪhæv",             ["AY", "HH", "AE", "V"]),
    ("v",                 ["V"]),
    ("uɹɛ",               ["UW", "R", "EH"]),
    ("pɪ",                ["P", "IH"]),
    ("ɪ",                 ["IH"]),
    ("bɑɾ",               ["B", "AA", "R"]),
    ("ə",                 ["AH"]),
    ("poʊnɑn",            ["P", "OW", "N", "AA", "N"]),
    ("siɪ",               ["S", "IY", "IH"]),
    ("eɪsɪnjus",          ["EY", "S", "IH", "N", "Y", "UW", "S"]),
    ("siŋðɪs",            ["S", "IY", "NG", "DH", "IH", "S"]),
]

passed = 0
failed = 0
for token, required in cases:
    result = _token_to_arpabet(token)
    # Each required phoneme must appear (order may differ due to parsing)
    missing = [p for p in required if p not in result]
    if not missing:
        print(f"  ✓  {token!r:20s} → {result}")
        passed += 1
    else:
        print(f"  ✗  {token!r:20s} → {result}   MISSING: {missing}")
        failed += 1

print(f"\n{passed} passed, {failed} failed out of {len(cases)} cases")
if failed == 0:
    print("\n✅ IPA parser is working correctly. Restart Streamlit and re-test.")
else:
    print("\n❌ Some tokens not parsed — paste this output for further fixes.")
