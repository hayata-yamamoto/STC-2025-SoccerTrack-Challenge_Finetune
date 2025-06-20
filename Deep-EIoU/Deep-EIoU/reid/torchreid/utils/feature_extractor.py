from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model

# SigLIP2用の新しいインポート
try:
    from transformers import AutoProcessor, AutoModel, AutoImageProcessor
    SIGLIP2_AVAILABLE = True
except ImportError:
    SIGLIP2_AVAILABLE = False
    print("Warning: transformers not available. SigLIP2 features disabled.")


class FeatureExtractor(object):
    """Enhanced feature extraction API with SigLIP2 support.

    Now supports both traditional ReID models and SigLIP2 vision-language models.

    Additional Args for SigLIP2:
        use_siglip2 (bool): Whether to use SigLIP2 instead of ReID models.
        siglip2_model (str): SigLIP2 model name from Hugging Face Hub.
        text_prompts (list): Text prompts for vision-language matching.
        return_image_features (bool): Whether to return only image features from SigLIP2.
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=False,
        # SigLIP2 specific parameters
        use_siglip2=False,
        siglip2_model='google/siglip2-base-patch16-224',
        text_prompts=None,
        return_image_features=True,
        quantization_config=None
    ):
        self.use_siglip2 = use_siglip2 and SIGLIP2_AVAILABLE
        self.return_image_features = return_image_features
        self.text_prompts = text_prompts or []

        if self.use_siglip2:
            # SigLIP2モデルの初期化
            self._init_siglip2(siglip2_model, device, quantization_config, verbose)
        else:
            # 既存のReIDモデルの初期化
            self._init_reid_model(
                model_name, model_path, image_size,
                pixel_mean, pixel_std, pixel_norm, device, verbose
            )

    def _init_siglip2(self, model_name, device, quantization_config, verbose):
        """SigLIP2モデルの初期化"""
        if not SIGLIP2_AVAILABLE:
            raise ImportError("transformers library is required for SigLIP2")

        # プロセッサーとモデルの読み込み
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        model_kwargs = {}
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
            # 量子化使用時はデータ型を自動設定
        else:
            # 量子化なしの場合は float32 を使用してデータ型の一貫性を保つ
            model_kwargs['torch_dtype'] = torch.float32
            if device.startswith('cuda'):
                model_kwargs['device_map'] = "auto"

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        if verbose:
            print(f'SigLIP2 Model: {model_name}')
            print(f'- Device: {next(self.model.parameters()).device}')
            print(f'- Dtype: {next(self.model.parameters()).dtype}')

        self.device = torch.device(device)
        if not quantization_config:
            self.model.to(self.device)

    def _init_reid_model(self, model_name, model_path, image_size,
                        pixel_mean, pixel_std, pixel_norm, device, verbose):
        """既存のReIDモデルの初期化（元のコード）"""
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, (1, 3, image_size[0], image_size[1])
            )
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)
        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input, text_prompts=None):
        """
        Args:
            input: Same as before (images)
            text_prompts: Optional text prompts for SigLIP2 (overrides default)
        """
        if self.use_siglip2:
            return self._extract_siglip2_features(input, text_prompts)
        else:
            return self._extract_reid_features(input)

    def _extract_siglip2_features(self, input, text_prompts=None):
        """SigLIP2による特徴抽出"""
        # 画像の前処理
        images = self._preprocess_images_siglip2(input)

        # テキストプロンプトの準備
        prompts = text_prompts or self.text_prompts

        if prompts and not self.return_image_features:
            # 画像-テキストマッチング
            # SigLIP2の推奨プロンプトテンプレートを使用
            texts = [f'This is a photo of {prompt}.' for prompt in prompts]
            inputs = self.processor(
                text=texts,
                images=images,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            )

            # デバイスとデータ型を適切に設定
            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
                    # モデルのデータ型に合わせる
                    model_dtype = next(self.model.parameters()).dtype
                    if inputs[key].dtype != model_dtype and key == 'pixel_values':
                        inputs[key] = inputs[key].to(model_dtype)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 類似度スコアを返す
                logits_per_image = outputs.logits_per_image
                return torch.sigmoid(logits_per_image)
        else:
            # 画像特徴量のみを抽出
            inputs = self.processor(images=images, return_tensors="pt")

            # デバイスとデータ型を適切に設定
            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
                    # モデルのデータ型に合わせる
                    model_dtype = next(self.model.parameters()).dtype
                    if inputs[key].dtype != model_dtype and key == 'pixel_values':
                        inputs[key] = inputs[key].to(model_dtype)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                return outputs

    def _extract_reid_features(self, input):
        """既存のReIDモデルによる特徴抽出（元のコード）"""
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features

    def _preprocess_images_siglip2(self, input):
        """SigLIP2用の画像前処理"""
        if isinstance(input, list):
            images = []
            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')
                elif isinstance(element, np.ndarray):
                    image = Image.fromarray(element).convert('RGB')
                else:
                    raise TypeError('Unsupported input type for SigLIP2')
                images.append(image)
            return images

        elif isinstance(input, str):
            return [Image.open(input).convert('RGB')]

        elif isinstance(input, np.ndarray):
            return [Image.fromarray(input).convert('RGB')]

        elif isinstance(input, torch.Tensor):
            # Convert tensor to PIL Image
            if input.dim() == 4:  # Batch of images
                images = []
                for i in range(input.size(0)):
                    img_tensor = input[i]
                    if img_tensor.max() <= 1.0:
                        img_tensor = (img_tensor * 255).byte()
                    img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
                    images.append(Image.fromarray(img_array).convert('RGB'))
                return images
            elif input.dim() == 3:  # Single image
                if input.max() <= 1.0:
                    input = (input * 255).byte()
                img_array = input.permute(1, 2, 0).cpu().numpy()
                return [Image.fromarray(img_array).convert('RGB')]

        raise NotImplementedError("Unsupported input format for SigLIP2")

    def set_text_prompts(self, prompts):
        """テキストプロンプトを設定"""
        self.text_prompts = prompts

    def get_similarity_scores(self, images, text_prompts):
        """画像とテキストの類似度スコアを取得"""
        if not self.use_siglip2:
            raise ValueError("Similarity scoring is only available with SigLIP2")

        old_return_mode = self.return_image_features
        self.return_image_features = False

        try:
            scores = self(images, text_prompts)
            return scores
        finally:
            self.return_image_features = old_return_mode
