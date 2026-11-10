import sys
sys.path.append('./')

from torch.utils.data import Dataset
import os
from PIL import Image, UnidentifiedImageError
from Dataset.image_cache import open_rgb_cached


class _SkipBadImageMixin:
    """Load images robustly: skip corrupt/empty/unreadable files and try neighbors."""

    def _open_rgb(self, path):
        # Prefer local SSD JPEG cache (see Config.DATA_CACHE_*); falls back to source.
        return open_rgb_cached(path)

    def __getitem__(self, index):
        n = len(self.data)
        if n == 0:
            raise RuntimeError(f'Empty dataset under {getattr(self, "root", "?")}')

        last_err = None
        # Walk forward from the requested index; wrap around at most once.
        for offset in range(n):
            i = (index + offset) % n
            path = self.data[i]
            target = self.target[i]
            if path in getattr(self, '_bad_paths', ()):
                continue
            try:
                img = self._open_rgb(path)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as e:
                if not hasattr(self, '_bad_paths'):
                    self._bad_paths = set()
                self._bad_paths.add(path)
                print(f'Warning: skip unreadable image ({e.__class__.__name__}): {path}')
                last_err = e
                continue

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        raise RuntimeError(
            f'No readable images in dataset (root={getattr(self, "root", "?")}); '
            f'last error: {last_err}'
        )


class genimage(_SkipBadImageMixin, Dataset):
    """GenImage / DiffusionFace layout: ``train|val/{nature,ai}``.

    Stores file paths only and loads images lazily in ``__getitem__``.
    Corrupt / empty / unreadable images are skipped at read time.
    """

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        noise_type=None,
        noise_rate=0,
        size=(256, 256),
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.size = size  # kept for API compatibility; Resize is usually in transform
        self._bad_paths = set()
        self.data, self.target = self.Construct_Participant_Dataset()

    def _list_image_paths(self, folder):
        if not os.path.isdir(folder):
            print(f'Warning: missing folder {folder}')
            return []
        paths = []
        for filename in sorted(os.listdir(folder)):
            file_path = os.path.join(folder, filename)
            if not os.path.isfile(file_path):
                continue
            # Skip empty files early (cheap); other corrupt files caught in __getitem__.
            try:
                if os.path.getsize(file_path) == 0:
                    print(f'Warning: skip empty file at scan: {file_path}')
                    continue
            except OSError:
                continue
            lower = filename.lower()
            if lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')):
                paths.append(file_path)
            elif '.' not in filename:
                paths.append(file_path)
        return paths

    def Construct_Participant_Dataset(self):
        if self.train:
            real_path = os.path.join(self.root, 'train', 'nature')
            fake_path = os.path.join(self.root, 'train', 'ai')
            tag_real, tag_fake = 'train_real_data', 'train_fake_data'
        else:
            real_path = os.path.join(self.root, 'val', 'nature')
            fake_path = os.path.join(self.root, 'val', 'ai')
            tag_real, tag_fake = 'test_real_data', 'test_fake_data'

        real_files = self._list_image_paths(real_path)
        fake_files = self._list_image_paths(fake_path)
        print(tag_real, real_path, len(real_files))
        print(tag_fake, fake_path, len(fake_files))

        data = real_files + fake_files
        labels = [0] * len(real_files) + [1] * len(fake_files)

        if self.dataidxs is not None:
            select_data = [data[i] for i in self.dataidxs]
            select_labels = [labels[i] for i in self.dataidxs]
        else:
            select_data = data
            select_labels = labels

        print('data.shape, labels.shape', len(select_data), len(select_labels))
        return select_data, select_labels

    def __len__(self):
        return len(self.data)


class celebdf2(_SkipBadImageMixin, Dataset):
    """Celeb-DF style layout with lazy path loading (same idea as genimage)."""

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        noise_type=None,
        noise_rate=0,
        size=(224, 224),
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.size = size
        self._bad_paths = set()
        self.data, self.target = self.Construct_Participant_Dataset()

    def _list_image_paths(self, folder):
        if not os.path.isdir(folder):
            print(f'Warning: missing folder {folder}')
            return []
        paths = []
        for filename in sorted(os.listdir(folder)):
            file_path = os.path.join(folder, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                if os.path.getsize(file_path) == 0:
                    print(f'Warning: skip empty file at scan: {file_path}')
                    continue
            except OSError:
                continue
            paths.append(file_path)
        return paths

    def Construct_Participant_Dataset(self):
        if self.train:
            real_path = os.path.join(self.root, 'train', 'real')
            fake_path = os.path.join(self.root, 'train', 'fake')
            tag_real, tag_fake = 'train_real_data', 'train_fake_data'
        else:
            real_path = os.path.join(self.root, 'test', 'test_real')
            fake_path = os.path.join(self.root, 'test', 'test_fake')
            tag_real, tag_fake = 'test_real_data', 'test_fake_data'

        real_files = self._list_image_paths(real_path)
        fake_files = self._list_image_paths(fake_path)
        print(tag_real, real_path, len(real_files))
        print(tag_fake, fake_path, len(fake_files))

        data = real_files + fake_files
        labels = [0] * len(real_files) + [1] * len(fake_files)

        if self.dataidxs is not None:
            select_data = [data[i] for i in self.dataidxs]
            select_labels = [labels[i] for i in self.dataidxs]
        else:
            select_data = data
            select_labels = labels

        print('data.shape, labels.shape', len(select_data), len(select_labels))
        return select_data, select_labels

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    noise_type = None
    noise_rate = 0
    dataset_root = '/home/mcsp/mcsp/ddim/data/genimage/adm'
    genim_train = genimage(root=dataset_root, noise_type=noise_type, noise_rate=noise_rate)
    genim_test = genimage(root=dataset_root, train=False)
    print(len(genim_train), len(genim_test), genim_train[0][0].size if hasattr(genim_train[0][0], 'size') else type(genim_train[0][0]))
