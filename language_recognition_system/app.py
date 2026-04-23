from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from language_recognition.inference import LanguagePredictor


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    app.config["CHECKPOINT_PATH"] = os.environ.get(
        "LID_CHECKPOINT",
        str(PROJECT_ROOT / "checkpoints" / "run_aug_fleurs240" / "best_model.pt"),
    )
    app.config["DEVICE"] = os.environ.get("LID_DEVICE", "cpu")

    predictor_cache: dict[str, LanguagePredictor] = {}

    def get_predictor() -> LanguagePredictor | None:
        checkpoint_path = app.config["CHECKPOINT_PATH"]
        if not Path(checkpoint_path).exists():
            return None
        cache_key = f"{checkpoint_path}:{app.config['DEVICE']}"
        predictor = predictor_cache.get(cache_key)
        if predictor is None:
            predictor = LanguagePredictor(checkpoint_path, device=app.config["DEVICE"])
            predictor_cache.clear()
            predictor_cache[cache_key] = predictor
        return predictor

    def _save_upload(file_obj) -> str:
        suffix = Path(secure_filename(file_obj.filename)).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_obj.save(tmp.name)
            return tmp.name

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = None
        predictor = get_predictor()

        if request.method == "POST":
            if predictor is None:
                error = "当前没有找到模型权重，请先训练模型或设置 LID_CHECKPOINT。"
            else:
                file = request.files.get("audio")
                if file is None or file.filename == "":
                    error = "请上传一个 .wav 音频文件。"
                elif not file.filename.lower().endswith(".wav"):
                    error = "当前页面只支持 .wav 文件。"
                else:
                    temp_path = _save_upload(file)
                    try:
                        result = predictor.predict_file(temp_path, top_k=3)
                    except Exception as exc:
                        error = f"识别失败：{exc}"
                    finally:
                        try:
                            Path(temp_path).unlink(missing_ok=True)
                        except OSError:
                            pass

        return render_template(
            "index.html",
            checkpoint_path=app.config["CHECKPOINT_PATH"],
            device=app.config["DEVICE"],
            model_ready=predictor is not None,
            result=result,
            error=error,
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """REST API 接口：上传 .wav 文件，返回 JSON 识别结果。

        请求：multipart/form-data，字段名 "audio"，值为 .wav 文件。
        响应示例：
          {"predicted_label": "zh", "predicted_score": 0.92, "top_k": [...]}
        错误响应：
          {"error": "描述信息"}，HTTP 状态码 4xx / 500。
        """
        predictor = get_predictor()
        if predictor is None:
            return jsonify({"error": "模型尚未就绪，请先训练或设置 LID_CHECKPOINT"}), 503

        file = request.files.get("audio")
        if file is None or file.filename == "":
            return jsonify({"error": "请在请求体中以 audio 字段上传音频文件"}), 400
        if not file.filename.lower().endswith(".wav"):
            return jsonify({"error": "当前接口只支持 .wav 文件"}), 415

        top_k = int(request.form.get("top_k", 3))
        temp_path = _save_upload(file)
        try:
            result = predictor.predict_file(temp_path, top_k=top_k)
        except Exception as exc:
            return jsonify({"error": f"识别失败：{exc}"}), 500
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
