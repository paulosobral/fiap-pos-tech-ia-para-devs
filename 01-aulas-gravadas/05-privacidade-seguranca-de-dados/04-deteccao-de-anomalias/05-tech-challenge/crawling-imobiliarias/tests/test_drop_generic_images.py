import importlib.util
import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("drop_generic_images", ROOT / "scripts" / "drop_generic_images.py")
dgi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dgi)


def blocks_image(seed: int, size=(640, 360)) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", size, "white")
    for _ in range(40):
        x, y = rng.randrange(size[0] - 60), rng.randrange(size[1] - 60)
        img.paste(tuple(rng.randrange(256) for _ in range(3)), (x, y, x + rng.randrange(20, 120), y + rng.randrange(20, 120)))
    return img


def letterboxed(img: Image.Image) -> Image.Image:
    framed = Image.new("RGB", (img.width, int(img.height * 1.5)), "white")
    framed.paste(img, (0, (framed.height - img.height) // 2))
    return framed


def test_same_image_in_other_format_is_detected_and_unrelated_is_not():
    generic = blocks_image(1)
    signature = dgi.gray_signature(generic)
    # mesmo conteúdo com barras brancas (como a variação 720x540) continua casando
    assert dgi.pearson(signature, dgi.gray_signature(letterboxed(generic))) > 0.9
    assert dgi.pearson(signature, dgi.gray_signature(blocks_image(2))) < 0.5


def test_shipped_signature_is_valid():
    signature, threshold = dgi.load_signature(ROOT / "scripts" / "generic_image_signature.json")
    assert len(signature) == 32 * 32 and 0.5 < threshold < 0.9
    assert dgi.pearson(signature, signature) > 0.999


def test_clean_drops_generic_placeholder_and_broken_but_keeps_real_and_unchecked():
    props = [
        {"id": "a", "images": ["https://cdn/x/real1.jpg?1", "https://cdn/x/colagem.jpg?2", "https://cdn/x/morta.jpg"]},
        {"id": "b", "images": ["https://cdn/x/colagem.jpg?3", "https://app.imoview.com.br//Front/img/house1.png"]},
        {"id": "c", "images": ["https://cdn/x/fora-do-ar.jpg"]},
        {"id": "d"},
    ]
    scores = {"https://cdn/x/real1.jpg": 0.2, "https://cdn/x/colagem.jpg": 0.97, "https://cdn/x/morta.jpg": dgi.BROKEN}
    out, stats = dgi.clean(props, lambda u: scores.get(u), threshold=0.65, workers=2)
    by_id = {p["id"]: p["images"] for p in out}
    assert by_id["a"] == ["https://cdn/x/real1.jpg?1"]
    assert by_id["b"] == []  # só colagem + ícone: sem foto é melhor que foto errada
    assert by_id["c"] == ["https://cdn/x/fora-do-ar.jpg"]  # erro de rede: na dúvida, mantém
    assert by_id["d"] == []
    assert (stats["dropped_generic"], stats["dropped_placeholder"], stats["dropped_broken"], stats["unchecked_kept"]) == (2, 1, 1, 1)


def test_cache_avoids_rescoring_and_keys_ignore_query_timestamp():
    calls = []
    scorer = lambda u: calls.append(u) or 0.1
    cache: dict[str, float] = {}
    props = [{"id": "a", "images": ["https://cdn/x/f.jpg?111"]}]
    dgi.clean(props, scorer, 0.65, cache, workers=1)
    dgi.clean([{"id": "a", "images": ["https://cdn/x/f.jpg?222"]}], scorer, 0.65, cache, workers=1)
    assert calls == ["https://cdn/x/f.jpg"]
