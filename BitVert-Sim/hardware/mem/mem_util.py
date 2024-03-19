from typing import Dict

class CactiUserConfig:
    def __init__(self, mode: str, option: Dict):
        self.mode = mode
        self.option = option

if __name__ == "__main__":
    config = CactiUserConfig('single', {'mem-type': 'ram', 'size': 128})
    print(config.mode)
    print(config.option)