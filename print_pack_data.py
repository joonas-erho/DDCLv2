#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Optional


@dataclass(frozen=True)
class ChartInfo:
    artist: str
    difficulty_name: str
    difficulty_rating: str


def _clean_sm_field(s: str) -> str:
    """
    StepMania .sm fields often contain trailing ':' or ';' and whitespace.
    This normalizes a single line field.
    """
    return s.strip().rstrip(":").rstrip(";").strip()


def _to_int_maybe(s: str) -> Optional[int]:
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None


def extract_charts_from_sm(sm_path: Path) -> list[ChartInfo]:
    """
    Extract charts from a .sm file by scanning for '#NOTES:' lines.

    Per user specification:
      - Find a line that says '#NOTES:'
      - Two lines after that is the chart artist
      - After that is the chart difficulty name
      - After that is the chart difficulty rating
    """
    try:
        lines = sm_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    charts: list[ChartInfo] = []

    for i, raw in enumerate(lines):
        if raw.strip() == "#NOTES:":
            artist_idx = i + 2
            diff_name_idx = i + 3
            diff_rating_idx = i + 4

            if diff_rating_idx >= len(lines):
                continue  # malformed/truncated block

            artist = _clean_sm_field(lines[artist_idx]) if artist_idx < len(lines) else ""
            difficulty_name = _clean_sm_field(lines[diff_name_idx]) if diff_name_idx < len(lines) else ""
            difficulty_rating = _clean_sm_field(lines[diff_rating_idx])

            # Basic sanity: skip empty parses
            if not (artist or difficulty_name or difficulty_rating):
                continue

            if difficulty_name == 'Edit':
                continue

            charts.append(
                ChartInfo(
                    artist=artist or "<unknown>",
                    difficulty_name=difficulty_name or "<unknown>",
                    difficulty_rating=difficulty_rating or "<unknown>",
                )
            )

    return charts


def iter_sm_files(pack_dir: Path) -> Iterable[Path]:
    # Search recursively within the pack directory for .sm files
    yield from pack_dir.rglob("*.sm")


def compute_pack_summary(
    sm_files: list[Path],
    charts_by_song: dict[str, list[ChartInfo]],
) -> tuple[set[str], int, str]:
    """
    Returns:
      (unique_chart_artists_set, song_count, total_difficulty_range_string)
    """
    all_charts = [c for charts in charts_by_song.values() for c in charts]
    unique_artists_set = {c.artist for c in all_charts}
    song_count = len(sm_files)

    numeric_ratings_all: list[int] = []
    for c in all_charts:
        r = _to_int_maybe(c.difficulty_rating)
        if r is not None:
            numeric_ratings_all.append(r)

    if numeric_ratings_all:
        total_range = f"{min(numeric_ratings_all)}-{max(numeric_ratings_all)}"
    else:
        total_range = "N/A"

    return unique_artists_set, song_count, total_range


def write_pack_report(
    pack_name: str,
    sm_files: list[Path],
    charts_by_song: dict[str, list[ChartInfo]],
    output_dir: Path,
) -> Path:
    all_charts = [c for charts in charts_by_song.values() for c in charts]
    unique_artists = {c.artist for c in all_charts}
    diff_distribution = Counter(c.difficulty_name for c in all_charts)

    # Counts like "Easy 2: 20"
    diff_and_rating_counts = Counter((c.difficulty_name, c.difficulty_rating) for c in all_charts)

    # Difficulty ranges
    numeric_ratings_all: list[int] = []
    numeric_ratings_by_diff: dict[str, list[int]] = defaultdict(list)
    for c in all_charts:
        r = _to_int_maybe(c.difficulty_rating)
        if r is None:
            continue
        numeric_ratings_all.append(r)
        numeric_ratings_by_diff[c.difficulty_name].append(r)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pack_name}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Songpack: {pack_name}\n")
        f.write(f"Songs (.sm files): {len(sm_files)}\n")
        f.write(f"Unique chart artists: {len(unique_artists)}\n")
        f.write("\n")

        f.write("Difficulty distribution (by difficulty name):\n")
        if diff_distribution:
            for name, count in diff_distribution.most_common():
                f.write(f"  - {name}: {count}\n")
        else:
            f.write("  (no charts found)\n")

        f.write("\n")
        f.write("Difficulty rankings counts (by difficulty name + rating):\n")
        if diff_and_rating_counts:
            # Sort by difficulty name, then numeric rating if possible, otherwise lexicographic
            def _sort_key(item: tuple[tuple[str, str], int]) -> tuple[str, int, str]:
                (diff_name, rating_str), _count = item
                rating_int = _to_int_maybe(rating_str)
                return (diff_name.lower(), rating_int if rating_int is not None else 10**9, rating_str)

            for (diff_name, rating_str), count in sorted(diff_and_rating_counts.items(), key=_sort_key):
                f.write(f"  - {diff_name} {rating_str}: {count}\n")
        else:
            f.write("  (no charts found)\n")

        f.write("\n")
        f.write("Difficulty ranges (min-max rating):\n")
        if numeric_ratings_all:
            f.write(f"  Total: {min(numeric_ratings_all)}-{max(numeric_ratings_all)}\n")
        else:
            f.write("  Total: (no numeric ratings found)\n")

        # Print per-difficulty ranges in a stable order
        for diff_name in sorted({c.difficulty_name for c in all_charts}, key=str.lower):
            ratings = numeric_ratings_by_diff.get(diff_name, [])
            if ratings:
                f.write(f"  {diff_name}: {min(ratings)}-{max(ratings)}\n")
            else:
                f.write(f"  {diff_name}: (no numeric ratings found)\n")

        f.write("\n")
        f.write("Per-song extracted ratings:\n")

        # Deterministic order for readability
        for song_key in sorted(charts_by_song.keys()):
            charts = charts_by_song[song_key]
            f.write(f"\n{song_key}\n")
            f.write("-" * len(song_key) + "\n")

            if not charts:
                f.write("  (no #NOTES: blocks parsed)\n")
                continue

            # Group by difficulty name
            grouped: dict[str, list[ChartInfo]] = defaultdict(list)
            for c in charts:
                grouped[c.difficulty_name].append(c)

            for diff_name in sorted(grouped.keys()):
                entries = grouped[diff_name]
                # show "artist: rating" pairs (keeps all ratings)
                pairs = ", ".join(f"{e.artist}: {e.difficulty_rating}" for e in entries)
                f.write(f"  {diff_name}: {pairs}\n")

    return out_path


def main() -> int:
    base_dir = Path("raw/songs")
    output_dir = Path("pack_reports")

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"ERROR: Base directory not found: {base_dir.resolve()}")
        return 2

    pack_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not pack_dirs:
        print(f"No pack folders found under: {base_dir.resolve()}")
        return 0

    packs_lines: list[str] = []
    total_songs_all_packs = 0
    unique_artists_all_packs: set[str] = set()

    for pack_dir in sorted(pack_dirs):
        pack_name = pack_dir.name
        sm_files = sorted(iter_sm_files(pack_dir))

        charts_by_song: dict[str, list[ChartInfo]] = {}

        for sm_path in sm_files:
            # A "song" here is counted as one .sm file, per your request.
            # Use a stable readable key: relative path under the pack.
            song_key = str(sm_path.relative_to(pack_dir))
            charts_by_song[song_key] = extract_charts_from_sm(sm_path)

        artists_set, song_count, total_range = compute_pack_summary(sm_files, charts_by_song)
        packs_lines.append(f"{pack_name} & {len(artists_set)} & {song_count} & {total_range} \\\\")
        total_songs_all_packs += song_count
        unique_artists_all_packs.update(artists_set)

        out_path = write_pack_report(pack_name, sm_files, charts_by_song, output_dir)
        print(f"Wrote: {out_path}")

    # Write combined packs summary
    output_dir.mkdir(parents=True, exist_ok=True)
    packs_path = output_dir / "packs.txt"
    packs_content = "\n".join(packs_lines)
    if packs_lines:
        packs_content += "\n\n"
    packs_content += f"Total songs: {total_songs_all_packs}\n"
    packs_content += f"Total unique chart artists: {len(unique_artists_all_packs)}\n"
    packs_path.write_text(packs_content, encoding="utf-8")
    print(f"Wrote: {packs_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())