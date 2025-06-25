import numpy as np
from collections import deque

from tracker import matching
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from collections import defaultdict


class PositionalIDManager:
    """位置情報に基づくシンプルなID管理"""

    def __init__(self, max_id=22):
        self.max_id = max_id
        self.active_ids = set()
        self.position_history = {}  # id -> [(frame, center_x, center_y, bbox_area), ...]
        self.occlusion_buffer = {}  # id -> 最後に見えた位置と特徴量

    def estimate_depth_order(self, tracks):
        """バウンディングボックスのサイズと位置から深度順序を推定"""
        if len(tracks) <= 1:
            return tracks

        # Y座標（画面下部）とボックスサイズで深度を推定
        # より下にいて、より大きいボックス = より手前（カメラに近い）
        depth_scores = []
        for track in tracks:
            center_x, center_y = (track.tlwh[0] + track.tlwh[2]/2, track.tlwh[1] + track.tlwh[3]/2)
            box_area = track.tlwh[2] * track.tlwh[3]

            # 深度スコア：y座標が大きく、ボックスが大きいほど手前
            depth_score = center_y * 0.7 + (box_area / 10000) * 0.3
            depth_scores.append((track, depth_score))

        # 深度順にソート（手前から奥へ）
        depth_scores.sort(key=lambda x: x[1], reverse=True)
        return [track for track, _ in depth_scores]

    def predict_position(self, track_id, frames_ahead=1):
        """位置履歴から次の位置を予測"""
        if track_id not in self.position_history or len(self.position_history[track_id]) < 2:
            return None

        history = self.position_history[track_id]
        recent = history[-2:]  # 最新2フレーム

        # 単純な線形予測
        dx = recent[1][1] - recent[0][1]  # x方向の変化
        dy = recent[1][2] - recent[0][2]  # y方向の変化

        pred_x = recent[1][1] + dx * frames_ahead
        pred_y = recent[1][2] + dy * frames_ahead

        return (pred_x, pred_y)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.last_tlwh = self._tlwh

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.features = []
        self.times = []
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))

        self.last_tlwh = new_track.tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.last_tlwh = new_tlwh

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
            self.times.append(frame_id)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def last_tlbr(self):
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Deep_EIoU(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # 最大track_id制限の設定
        max_id = getattr(args, 'max_track_id', 22)
        BaseTrack.set_max_track_id(max_id)

        # シンプルな位置ベースID管理
        self.pos_id_manager = PositionalIDManager(max_id)
        self.enable_position_tracking = getattr(args, 'enable_position_tracking', True)
        self.position_weight = getattr(args, 'position_weight', 0.7)
        self.occlusion_recovery_frames = getattr(args, 'occlusion_recovery_frames', 30)

        # ID修正メカニズム用のパラメータ
        self.id_correction_enabled = getattr(args, 'enable_id_correction', True)
        self.correction_buffer_size = getattr(args, 'correction_buffer_size', 10)
        self.correction_thresh = getattr(args, 'correction_thresh', 0.3)
        self.track_history = defaultdict(list)  # track_id -> list of features
        self.appearance_history = defaultdict(list)  # track_id -> list of (frame_id, feature)

    def update(self, output_results, embedding):

        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:,4]
                bboxes = output_results[:, :4]  # x1y1x2y2
            elif output_results.shape[1] == 7:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                # import pdb;pdb.set_trace()
            else:
                raise ValueError('Wrong detection size {}'.format(output_results.shape[1]))


            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]

            if self.args.with_reid:
                embedding = embedding[lowest_inds]
                features_keep = embedding[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Associate with high score detection boxes
        num_iteration = 2
        init_expand_scale = 0.7
        expand_scale_step = 0.1

        for iteration in range(num_iteration):

            cur_expand_scale = init_expand_scale + expand_scale_step*iteration

            ious_dists = matching.eiou_distance(strack_pool, detections, cur_expand_scale)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if self.args.with_reid:
                # サッカー向けの強化されたReID距離計算を使用
                use_enhanced_reid = getattr(self.args, 'enhanced_reid', True)
                if use_enhanced_reid:
                    emb_dists = matching.enhanced_embedding_distance(strack_pool, detections) / 2.0
                else:
                    emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)
            else:
                dists = ious_dists

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            if self.args.with_reid:
                features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                                    (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                    (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if self.args.with_reid:
            use_enhanced_reid = getattr(self.args, 'enhanced_reid', True)
            if use_enhanced_reid:
                emb_dists = matching.enhanced_embedding_distance(unconfirmed, detections) / 2.0
            else:
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 位置ベース管理：新規検出に対するスマートID割り当て
        unmatched_detections = [detections[i] for i in u_detection]
        print(f"[DEBUG] Step 4: enable_position_tracking={self.enable_position_tracking}, unmatched_detections={len(unmatched_detections)}")

        if self.enable_position_tracking and unmatched_detections:
            print(f"[DEBUG] Using position-based tracking for {len(unmatched_detections)} detections")
            for detection in unmatched_detections:
                if detection.score < self.new_track_thresh:
                    continue

                det_center = (detection.tlwh[0] + detection.tlwh[2]/2,
                             detection.tlwh[1] + detection.tlwh[3]/2)
                print(f"[DEBUG] Processing detection at center ({det_center[0]:.1f}, {det_center[1]:.1f})")

                # オクルージョンからの復帰をチェック
                recovered_id = self.check_occlusion_recovery(detection, det_center)

                if recovered_id:
                    # 既存IDを再利用
                    print(f"[DEBUG] Reusing recovered ID: {recovered_id}")
                    detection.track_id = recovered_id
                    detection.activate(self.kalman_filter, self.frame_id)
                    activated_starcks.append(detection)
                else:
                    # 新規ID割り当て（上限チェック付き）
                    new_id = self.assign_new_id_with_limit()
                    if new_id:
                        print(f"[DEBUG] Creating new track with ID: {new_id}")
                        # 一時的にtrack_idを設定してからactivate
                        BaseTrack._count = new_id - 1  # activateでnext_id()が呼ばれるため
                        detection.activate(self.kalman_filter, self.frame_id)
                        activated_starcks.append(detection)
                    else:
                        print(f"[DEBUG] Failed to assign new ID - skipping detection")
        else:
            # 従来の方法
            print(f"[DEBUG] Using traditional tracking for {len(u_detection)} detections")
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.new_track_thresh:
                    continue

                print(f"[DEBUG] Creating track with traditional method, ID will be: {BaseTrack._count + 1}")
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 位置ベース管理の実行
        if self.enable_position_tracking:
            print(f"[DEBUG] Frame {self.frame_id}: Running position-based management")
            print(f"[DEBUG] Current tracked tracks: {len(self.tracked_stracks)}, lost tracks: {len(self.lost_stracks)}")

            # 位置履歴を更新
            self.update_position_history()

            # オクルージョン処理
            self.handle_occlusions()

            # 現在のIDの状況を表示
            current_ids = [t.track_id for t in self.tracked_stracks]
            print(f"[DEBUG] Current track IDs: {sorted(current_ids)}")
            print(f"[DEBUG] Active IDs in manager: {sorted(self.pos_id_manager.active_ids)}")
            print(f"[DEBUG] Occlusion buffer: {list(self.pos_id_manager.occlusion_buffer.keys())}")

            # ID上限チェック
            max_current_id = max(current_ids) if current_ids else 0
            if max_current_id > self.pos_id_manager.max_id:
                print(f"[WARNING] ID {max_current_id} exceeds limit {self.pos_id_manager.max_id}!")
        else:
            print(f"[DEBUG] Position-based management is disabled")

        # ID修正メカニズムの実行
        if self.id_correction_enabled and self.args.with_reid:
            self.perform_id_correction()

        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks

    def perform_id_correction(self):
        """オクルージョン後のID修正を実行"""
        if len(self.tracked_stracks) < 2:
            return

        # 外観履歴を更新
        for track in self.tracked_stracks:
            if track.smooth_feat is not None:
                self.appearance_history[track.track_id].append(
                    (self.frame_id, track.smooth_feat.copy())
                )
                # 古い履歴を削除
                if len(self.appearance_history[track.track_id]) > self.correction_buffer_size:
                    self.appearance_history[track.track_id].pop(0)

        # ID入れ替わりの検出と修正
        track_pairs = []
        for i, track_a in enumerate(self.tracked_stracks):
            for j, track_b in enumerate(self.tracked_stracks[i+1:], i+1):
                track_pairs.append((track_a, track_b))

        for track_a, track_b in track_pairs:
            if self.should_swap_ids(track_a, track_b):
                # print(f"[ID Correction] Swapping IDs: {track_a.track_id} <-> {track_b.track_id}")
                self.swap_track_ids(track_a, track_b)

    def should_swap_ids(self, track_a, track_b):
        """2つのトラックのIDを入れ替えるべきかを判定"""
        if (track_a.track_id not in self.appearance_history or
            track_b.track_id not in self.appearance_history):
            return False

        history_a = self.appearance_history[track_a.track_id]
        history_b = self.appearance_history[track_b.track_id]

        if len(history_a) < 3 or len(history_b) < 3:
            return False

        # 最近の特徴量と過去の特徴量を比較
        recent_feat_a = track_a.smooth_feat
        recent_feat_b = track_b.smooth_feat

        # track_aの最近の特徴量がtrack_bの過去の特徴量により近いかチェック
        past_features_a = [feat for _, feat in history_a[:-2]]  # 最近2つを除く
        past_features_b = [feat for _, feat in history_b[:-2]]

        if len(past_features_a) == 0 or len(past_features_b) == 0:
            return False

        # コサイン類似度を計算
        from scipy.spatial.distance import cosine

        # track_aの現在の特徴量 vs track_bの過去の特徴量
        sim_a_to_past_b = np.mean([1 - cosine(recent_feat_a, feat)
                                   for feat in past_features_b])

        # track_bの現在の特徴量 vs track_aの過去の特徴量
        sim_b_to_past_a = np.mean([1 - cosine(recent_feat_b, feat)
                                   for feat in past_features_a])

        # track_aの現在の特徴量 vs track_aの過去の特徴量
        sim_a_to_past_a = np.mean([1 - cosine(recent_feat_a, feat)
                                   for feat in past_features_a])

        # track_bの現在の特徴量 vs track_bの過去の特徴量
        sim_b_to_past_b = np.mean([1 - cosine(recent_feat_b, feat)
                                   for feat in past_features_b])

        # 入れ替えるべき条件：
        # 1. 現在のAが過去のBにより類似している
        # 2. 現在のBが過去のAにより類似している
        # 3. 類似度の改善が閾値を超えている

        improvement = (sim_a_to_past_b + sim_b_to_past_a) - (sim_a_to_past_a + sim_b_to_past_b)

        return improvement > self.correction_thresh

    def swap_track_ids(self, track_a, track_b):
        """2つのトラックのIDを入れ替える"""
        temp_id = track_a.track_id
        track_a.track_id = track_b.track_id
        track_b.track_id = temp_id

        # 履歴も入れ替える
        temp_history = self.appearance_history[temp_id]
        self.appearance_history[track_a.track_id] = self.appearance_history[track_b.track_id]
        self.appearance_history[track_b.track_id] = temp_history

    def check_occlusion_recovery(self, detection, det_center):
        """オクルージョンからの復帰をチェック"""
        best_match_id = None
        best_score = 0

        print(f"[DEBUG] Checking occlusion recovery. Buffer size: {len(self.pos_id_manager.occlusion_buffer)}")

        for track_id, buffer_data in self.pos_id_manager.occlusion_buffer.items():
            if track_id in self.pos_id_manager.active_ids:
                continue

            last_pos, last_feature, last_frame = buffer_data

            # 1. 位置的近さをチェック
            pos_distance = np.sqrt((det_center[0] - last_pos[0])**2 +
                                 (det_center[1] - last_pos[1])**2)

            # 2. 予測位置との近さをチェック
            frames_gap = self.frame_id - last_frame
            predicted_pos = self.pos_id_manager.predict_position(track_id, frames_gap)

            if predicted_pos:
                pred_distance = np.sqrt((det_center[0] - predicted_pos[0])**2 +
                                      (det_center[1] - predicted_pos[1])**2)
                pos_score = max(0, 1 - pred_distance / 200)  # 200ピクセル以内で高スコア
            else:
                pos_score = max(0, 1 - pos_distance / 150)

            # 3. 特徴量の類似度（利用可能な場合）
            if detection.curr_feat is not None and last_feature is not None:
                from scipy.spatial.distance import cosine
                feature_sim = 1 - cosine(detection.curr_feat, last_feature)
                feature_score = max(0, feature_sim)
            else:
                feature_score = 0.5  # デフォルトスコア

            # 4. 総合スコア（位置を重視）
            total_score = pos_score * self.position_weight + feature_score * (1 - self.position_weight)

            print(f"[DEBUG] Track {track_id}: pos_score={pos_score:.3f}, feature_score={feature_score:.3f}, total_score={total_score:.3f}")

            if total_score > best_score and total_score > 0.6:  # 閾値
                best_score = total_score
                best_match_id = track_id

        if best_match_id:
            print(f"[DEBUG] ID Recovery: {best_match_id} with score {best_score:.3f}")
        else:
            print(f"[DEBUG] No ID recovery found")

        return best_match_id

    def assign_new_id_with_limit(self):
        """上限付きの新規ID割り当て"""
        print(f"[DEBUG] Assigning new ID. Active IDs: {len(self.pos_id_manager.active_ids)}, Max: {self.pos_id_manager.max_id}")
        print(f"[DEBUG] Current active IDs: {sorted(self.pos_id_manager.active_ids)}")

        # 使用可能なIDを探す
        for candidate_id in range(1, self.pos_id_manager.max_id + 1):
            if candidate_id not in self.pos_id_manager.active_ids:
                self.pos_id_manager.active_ids.add(candidate_id)
                print(f"[DEBUG] Assigned new ID: {candidate_id}")
                return candidate_id

        # IDが不足している場合、最も古い・低品質なトラックを削除
        print(f"[DEBUG] ID limit reached! Attempting to recycle...")
        return self.recycle_id_from_low_quality_track()

    def recycle_id_from_low_quality_track(self):
        """低品質なトラックからIDを回収"""
        if not self.tracked_stracks and not self.lost_stracks:
            print(f"[DEBUG] No tracks to recycle from")
            return None

        # 候補：短時間・低スコア・動きが少ないトラック
        candidates = []
        for track in self.tracked_stracks + self.lost_stracks:
            quality_score = (track.tracklet_len * 0.4 +
                           track.score * 0.4 +
                           (self.frame_id - track.start_frame) * 0.2)
            candidates.append((track, quality_score))

        if candidates:
            # 最低品質のトラックを削除
            worst_track, quality = min(candidates, key=lambda x: x[1])
            recycled_id = worst_track.track_id

            print(f"[DEBUG] Recycling ID {recycled_id} from track with quality {quality:.3f}")

            # トラックを削除
            if worst_track in self.tracked_stracks:
                self.tracked_stracks.remove(worst_track)
            if worst_track in self.lost_stracks:
                self.lost_stracks.remove(worst_track)

            self.pos_id_manager.active_ids.discard(recycled_id)
            return recycled_id

        print(f"[DEBUG] No candidates found for recycling")
        return None

    def update_position_history(self):
        """位置履歴を更新"""
        for track in self.tracked_stracks:
            center_x = track.tlwh[0] + track.tlwh[2] / 2
            center_y = track.tlwh[1] + track.tlwh[3] / 2
            box_area = track.tlwh[2] * track.tlwh[3]

            if track.track_id not in self.pos_id_manager.position_history:
                self.pos_id_manager.position_history[track.track_id] = []

            self.pos_id_manager.position_history[track.track_id].append(
                (self.frame_id, center_x, center_y, box_area)
            )

            # 履歴の長さを制限
            if len(self.pos_id_manager.position_history[track.track_id]) > 30:
                self.pos_id_manager.position_history[track.track_id].pop(0)

            # アクティブIDを更新
            self.pos_id_manager.active_ids.add(track.track_id)

    def handle_occlusions(self):
        """オクルージョンの処理"""
        # 消失したトラックをバッファに保存
        for track in self.lost_stracks:
            if track.track_id not in self.pos_id_manager.occlusion_buffer:
                center_x = track.last_tlwh[0] + track.last_tlwh[2] / 2
                center_y = track.last_tlwh[1] + track.last_tlwh[3] / 2

                print(f"[DEBUG] Adding track {track.track_id} to occlusion buffer at position ({center_x:.1f}, {center_y:.1f})")

                self.pos_id_manager.occlusion_buffer[track.track_id] = (
                    (center_x, center_y),
                    track.smooth_feat.copy() if track.smooth_feat is not None else None,
                    track.frame_id
                )

        # 古いバッファを削除
        current_frame = self.frame_id
        expired_ids = []
        for track_id, (_, _, last_frame) in self.pos_id_manager.occlusion_buffer.items():
            if current_frame - last_frame > self.occlusion_recovery_frames:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            print(f"[DEBUG] Removing expired track {track_id} from occlusion buffer")
            del self.pos_id_manager.occlusion_buffer[track_id]
            self.pos_id_manager.active_ids.discard(track_id)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
