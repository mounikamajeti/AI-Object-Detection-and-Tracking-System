class ObjectCounter:
    def __init__(self):
        self.active_tracks: dict[int, bool] = {}
        self.entered = 0
        self.left = 0

    def update(self, current_ids: list[int]) -> None:
        new_ids = set(current_ids) - set(self.active_tracks.keys())
        for _ in new_ids:
            self.entered += 1
        left_ids = set(self.active_tracks.keys()) - set(current_ids)
        for _ in left_ids:
            self.left += 1
        self.active_tracks = {track_id: True for track_id in current_ids}

    def get_total(self) -> int:
        return self.entered

    def get_current(self) -> int:
        return len(self.active_tracks)

    def get_left(self) -> int:
        return self.left
