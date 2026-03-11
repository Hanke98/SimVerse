import os
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def _convert_task(args):
    path, fname, frame_id, new_name = args

    src = os.path.join(path, fname)
    dst_name = f"{new_name}_{frame_id:06d}.png"
    dst = os.path.join(path, dst_name)

    with Image.open(src) as img:
        img.save(dst)

    os.remove(src)

    return f"{fname} -> {dst_name}"


def BMP2PNG(path, old_name, new_name, num_threads=8):
    pattern = re.compile(rf"{old_name}_(\d+)\.bmp")

    files = []
    for fname in os.listdir(path):
        m = pattern.match(fname)
        if m:
            frame_id = int(m.group(1))
            files.append((frame_id, fname))

    files.sort(key=lambda x: x[0])

    tasks = [(path, fname, frame_id, new_name) for frame_id, fname in files]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for result in executor.map(_convert_task, tasks):
            print(result)


if __name__ == '__main__':
    path = os.getcwd()
    BMP2PNG(path, 'screen_capture', 'Frame')