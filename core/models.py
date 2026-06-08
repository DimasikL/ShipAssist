import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class CommonSearchModel:
    search_commands: List[str]
    verbose: bool

    def verbose_print(self, *args):
        if self.verbose:
            print(*args)

    def search_keywords(self, wav_path) -> Optional[str]:
        raise Exception("Implement me")
