# pianopipe

Turns piano MIDI files into practice-ready songs for [Quest Piano Trainer](https://github.com/JeremyAley/quest-piano-catalog), a mixed-reality piano trainer for Meta Quest 3, and builds the `catalog.json` the app reads.

For each input file it:

- picks the piano part (General MIDI pianos, drums dropped) and finds the hands (track names, two tracks, two channels, or an automatic split);
- writes four difficulty levels (1 Melody, 2 Melody + bass, 3 Simplified, 4 Full) as app-ready MIDI: track 1 right hand, track 2 left hand;
- adds a suggested finger (1-5) to every note, as text events `f1`-`f5` before each note-on, from its own cost model (no learned data);
- rates each level: a difficulty number (1-10) and an estimated grade (0 Initial, 1-8 exam grades, 9-10 diploma);
- checks playability and skips files whose hands can't be trusted;
- writes `catalog.json` (format version 1) with a SHA-256 per file.

The same pipeline builds the public catalog ([quest-piano-catalog](https://github.com/JeremyAley/quest-piano-catalog)) and the private song libraries players keep in their own repositories.

## Install

```bash
pip install "git+https://github.com/JeremyAley/pianopipe@v0.3.0"
```

Python 3.10 or later; the only dependency is [mido](https://github.com/mido/mido). `pianopipe --version` prints the installed version.

## Use

```bash
pianopipe inspect song.mid                    # tracks, channels, programs, which hands were found
pianopipe process song.mid --out out/         # one song: out/<id>/level1..4.mid and its catalog entry
pianopipe build songs/ out/catalog --name "My library"   # a folder of MIDI files -> catalog.json + levels
pianopipe eval-fingering                      # fingering against textbook cases
```

## Develop

```bash
pip install -e ".[dev]"
pytest -q
```

## Licence

Apache License 2.0, see [LICENSE](LICENSE). The code only: songs you process keep their own licences, and you are responsible for having the right to use them.
