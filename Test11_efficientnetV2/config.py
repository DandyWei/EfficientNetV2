from pathlib import Path
import os


#
# Path.cwd()
# train_pths = [p for p in (Path.cwd() / Path('Data') / Path('Train') ).glob('*/*.npy')]
class config:
    dir_path = Path('..')
    _train_fps    = [p for p in ( dir_path / Path('Data') / Path('Train') ).glob('*/*.npy')]
    _test_fps     = [p for p in ( dir_path / Path('Data') / Path('Train') ).glob('*/*.npy')]

    _train_labels = [p.parent.name for p in _train_fps]
    _test_labels  = [p.parent.name for p in _test_fps]


    @property
    def train_pths(self, ) -> list:
        return self._train_fps
    @property
    def test_pths(self, ):
        return self._train_fps
    @property
    def train_labels(self, ):
        return self._train_labels
    @property
    def test_labels(self, ):
        return self._test_labels

if __name__ == "__main__":
    cf = config()
    print(cf.test_labels)
