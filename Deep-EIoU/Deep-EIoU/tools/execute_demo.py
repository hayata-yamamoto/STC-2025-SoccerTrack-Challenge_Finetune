#!/usr/bin/env python3
import subprocess
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_demo_config(config):
    """単一設定でdemo.pyを実行"""
    fp = "/content/Deep-EIoU/Deep-EIoU/117092.mp4"

    cmd = [
        "python3.8", "tools/demo.py",
        "--path", fp,
        "--detector", "yolov11",
        "--det_ckpt", "./best.pt",
        "--experiment-name", config["name"]
    ]

    # 設定パラメータを追加
    for key, value in config["params"].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    print(f"Starting: {config['name']}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2時間タイムアウト
        end_time = time.time()

        return {
            "name": config["name"],
            "success": result.returncode == 0,
            "duration": end_time - start_time,
            "stdout": result.stdout[-1000:] if result.stdout else "",  # 最後の1000文字のみ
            "stderr": result.stderr[-1000:] if result.stderr else ""
        }
    except subprocess.TimeoutExpired:
        return {
            "name": config["name"],
            "success": False,
            "duration": time.time() - start_time,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "name": config["name"],
            "success": False,
            "duration": time.time() - start_time,
            "error": str(e)
        }

def main():
    configs = [
        # === 既存の6設定 ===
        {
            "name": "11709_yolo",
            "params": {
                "det_conf": 0.5,
                "track_high_thresh": 0.7,
                "match_thresh": 0.9,
                "use_siglip2": False,
                "new_track_thresh": 0.8
            }
        },
        {
            "name": "117092_high_precision",
            "params": {
                "det_conf": 0.5,
                "track_high_thresh": 0.7,
                "match_thresh": 0.9,
                "new_track_thresh": 0.8
            }
        },
        {
            "name": "117092_balanced",
            "params": {
                "det_conf": 0.3,
                "track_high_thresh": 0.6,
                "match_thresh": 0.8,
                "new_track_thresh": 0.7
            }
        },
        {
            "name": "117092_high_detection",
            "params": {
                "det_conf": 0.1,
                "track_high_thresh": 0.4,
                "match_thresh": 0.7,
                "new_track_thresh": 0.5
            }
        },
        {
            "name": "117092_id_correction",
            "params": {
                "det_conf": 0.2,
                "enable_id_correction": True
            }
        },
        {
            "name": "117092_soccer_enhanced",
            "params": {
                "det_conf": 0.2,
                "enhanced_reid": True,
                "enable_id_correction": True,
                "enable_soccer_preprocessing": True,
                "enhance_contrast": True,
                "gamma_correction": True
            }
        },

        # === 追加の12設定 ===

        # 超高検出感度設定
        {
            "name": "117092_ultra_sensitive",
            "params": {
                "det_conf": 0.05,
                "track_high_thresh": 0.3,
                "match_thresh": 0.6,
                "new_track_thresh": 0.4,
                "track_buffer": 90
            }
        },

        # 中間検出設定 + SigLIP2無効
        {
            "name": "117092_medium_no_siglip",
            "params": {
                "det_conf": 0.25,
                "track_high_thresh": 0.5,
                "match_thresh": 0.75,
                "new_track_thresh": 0.6,
                "use_siglip2": False
            }
        },

        # 長期追跡重視設定
        {
            "name": "117092_long_track",
            "params": {
                "det_conf": 0.15,
                "track_high_thresh": 0.5,
                "match_thresh": 0.85,
                "new_track_thresh": 0.7,
                "track_buffer": 120,
                "proximity_thresh": 0.6
            }
        },

        # ReID重視設定
        {
            "name": "117092_reid_focused",
            "params": {
                "det_conf": 0.2,
                "track_high_thresh": 0.6,
                "match_thresh": 0.95,
                "appearance_thresh": 0.15,
                "proximity_thresh": 0.3,
                "enhanced_reid": True
            }
        },

        # 高速追跡設定（短いバッファ）
        {
            "name": "117092_fast_track",
            "params": {
                "det_conf": 0.35,
                "track_high_thresh": 0.8,
                "match_thresh": 0.8,
                "new_track_thresh": 0.9,
                "track_buffer": 30
            }
        },

        # ID補正 + 中精度設定
        {
            "name": "117092_id_correct_medium",
            "params": {
                "det_conf": 0.25,
                "track_high_thresh": 0.55,
                "match_thresh": 0.8,
                "enable_id_correction": True,
                "correction_buffer_size": 15,
                "correction_thresh": 0.25
            }
        },

        # サッカー前処理のみ
        {
            "name": "117092_soccer_preprocess_only",
            "params": {
                "det_conf": 0.2,
                "enable_soccer_preprocessing": True,
                "enhance_contrast": True,
                "denoise": True,
                "use_field_roi": True
            }
        },

        # 厳格設定（高閾値全般）
        {
            "name": "117092_strict",
            "params": {
                "det_conf": 0.6,
                "track_high_thresh": 0.8,
                "match_thresh": 0.95,
                "new_track_thresh": 0.9,
                "appearance_thresh": 0.35
            }
        },

        # 寛容設定（低閾値全般）
        {
            "name": "117092_lenient",
            "params": {
                "det_conf": 0.08,
                "track_high_thresh": 0.35,
                "match_thresh": 0.65,
                "new_track_thresh": 0.45,
                "appearance_thresh": 0.2
            }
        },

        # ガンマ補正特化
        {
            "name": "117092_gamma_enhanced",
            "params": {
                "det_conf": 0.2,
                "enable_soccer_preprocessing": True,
                "gamma_correction": True,
                "adaptive_detection": True,
                "track_high_thresh": 0.55
            }
        },

        # バランス + ID補正
        {
            "name": "117092_balanced_id_correct",
            "params": {
                "det_conf": 0.3,
                "track_high_thresh": 0.6,
                "match_thresh": 0.8,
                "new_track_thresh": 0.7,
                "enable_id_correction": True,
                "correction_thresh": 0.35
            }
        },

        # 全機能ON設定
        {
            "name": "117092_full_features",
            "params": {
                "det_conf": 0.2,
                "enhanced_reid": True,
                "enable_id_correction": True,
                "enable_soccer_preprocessing": True,
                "enhance_contrast": True,
                "gamma_correction": True,
                "denoise": True,
                "adaptive_detection": True,
                "correction_buffer_size": 12
            }
        }
    ]
    print(f"Starting {len(configs)} parallel demo.py executions...")

    # 並列実行（最大同時実行数を制限）
    max_workers = min(len(configs), 10)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 全タスクを送信
        future_to_config = {executor.submit(run_demo_config, config): config for config in configs}

        # 完了したものから結果を処理
        for future in as_completed(future_to_config):
            result = future.result()
            config = future_to_config[future]

            if result["success"]:
                print(f"✅ {result['name']} completed in {result['duration']:.1f}s")
            else:
                print(f"❌ {result['name']} failed after {result['duration']:.1f}s")
                if "error" in result:
                    print(f"   Error: {result['error']}")
                if result.get("stderr"):
                    print(f"   Stderr: {result['stderr'][:200]}...")

    print("All parallel executions completed!")

if __name__ == "__main__":
    main()