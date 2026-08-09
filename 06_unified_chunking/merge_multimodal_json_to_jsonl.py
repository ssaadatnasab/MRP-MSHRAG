#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Iterable, List

BASE_DIR = Path('/home/user/Desktop/Markdown_Outputs_Stage7')
FOLDERS = ['Formula', 'Image', 'Table']


def _iter_chunks(obj: Any) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ('chunks', 'items', 'documents', 'data', 'records'):
            value = obj.get(key)
            if isinstance(value, list):
                return value
        return [obj]
    return [obj]


def convert_folder(folder_name: str) -> Path:
    folder = BASE_DIR / folder_name
    output_path = folder / f'{folder_name}.jsonl'

    if not folder.exists():
        raise FileNotFoundError(f'Folder not found: {folder}')

    json_files = sorted(
        p for p in folder.rglob('*.json')
        if p.is_file() and p.name != output_path.name
    )

    written = 0
    with output_path.open('w', encoding='utf-8') as out_f:
        for json_path in json_files:
            with json_path.open('r', encoding='utf-8') as in_f:
                try:
                    payload = json.load(in_f)
                except json.JSONDecodeError as exc:
                    print(f'Skipping invalid JSON: {json_path} ({exc})')
                    continue

            chunks = _iter_chunks(payload)
            for chunk in chunks:
                if isinstance(chunk, dict):
                    out_f.write(json.dumps(chunk, ensure_ascii=False))
                else:
                    out_f.write(json.dumps({'value': chunk}, ensure_ascii=False))
                out_f.write('\n')
                written += 1

    print(f'Created {output_path} with {written} chunks from {len(json_files)} JSON files')
    return output_path


def main() -> None:
    for folder_name in FOLDERS:
        convert_folder(folder_name)


if __name__ == '__main__':
    main()
