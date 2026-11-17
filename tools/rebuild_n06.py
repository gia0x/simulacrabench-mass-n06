"""Rebuild the N06 archive and verify its SHA-256."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory with manifest.json and source/")
    parser.add_argument("output", type=Path, help="New ZIP path")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_bytes())

    # Match the metadata, file order and compression of tools/build_submission.py.
    payload = {}
    for name, expected in sorted(manifest["payload_sha256"].items()):
        data = (args.run / "source" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            parser.error(f"Frozen source checksum differs for {name}")
        payload[name] = data

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in payload.items():
            member = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    result = buffer.getvalue()
    digest = hashlib.sha256(result).hexdigest()
    if digest != manifest["submission_sha256"]:
        parser.error(f"Rebuilt ZIP checksum differs from manifest: {digest}")
    with args.output.open("xb") as file:
        file.write(result)
    print(f"{digest}  {args.output}")


if __name__ == "__main__":
    main()
