import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(object):
    _count = 0
    _available_ids = []  # setではなくlistを使用して順序を保持
    _max_track_id = None  # 最大track_id制限

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        # デバッグ情報を出力（必要に応じてコメントアウト）
        # print(f"[DEBUG] next_id called: available_ids={BaseTrack._available_ids}, _count={BaseTrack._count}")

        # 再利用可能なIDがある場合は、最小のIDを使用（予測可能性を向上）
        if BaseTrack._available_ids:
            BaseTrack._available_ids.sort()  # 昇順ソート
            selected_id = BaseTrack._available_ids.pop(0)  # 最小のIDを取得
            # print(f"[DEBUG] Reusing ID: {selected_id}")
            return selected_id

        # 最大track_id制限がある場合のチェック
        if BaseTrack._max_track_id is not None and BaseTrack._count >= BaseTrack._max_track_id:
            # 制限に達した場合は1から再開（循環利用）
            BaseTrack._count = 0

        BaseTrack._count += 1
        # print(f"[DEBUG] New ID generated: {BaseTrack._count}")
        return BaseTrack._count

    @staticmethod
    def return_id(track_id):
        """削除されたtrack_idを再利用可能なプールに戻す"""
        if track_id > 0 and track_id not in BaseTrack._available_ids:  # 重複チェックを追加
            BaseTrack._available_ids.append(track_id)
            # print(f"[DEBUG] ID {track_id} returned to pool. Available IDs: {sorted(BaseTrack._available_ids)}")

    @staticmethod
    def set_max_track_id(max_id):
        """最大track_id制限を設定"""
        BaseTrack._max_track_id = max_id

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed
        # track_idを再利用可能なプールに戻す
        BaseTrack.return_id(self.track_id)

    @staticmethod
    def clear_count():
        BaseTrack._count = 0
        BaseTrack._available_ids.clear()

    @staticmethod
    def enable_debug():
        """デバッグ出力を有効にする"""
        import sys
        # この関数が呼ばれた時点で、デバッグコメントを有効化するフラグとして使用可能
        BaseTrack._debug_enabled = True

    @staticmethod
    def get_debug_info():
        """現在のID管理状態をデバッグ用に取得"""
        return {
            'count': BaseTrack._count,
            'available_ids': sorted(BaseTrack._available_ids),
            'max_track_id': BaseTrack._max_track_id
        }
