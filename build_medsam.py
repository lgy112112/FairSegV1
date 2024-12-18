import torch
import torch.nn as nn
import torch.nn.functional as F


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


class Discriminator(nn.Sequential):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_size * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_size * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_size * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_size * 8, num_classes)
        )


class Predictor(nn.Sequential):
    def __init__(self, num_classes, hidden_size=768):
        super().__init__(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        num_dim = len(x.shape)
        if num_dim == 4:
            x = torch.flatten(x, start_dim=2, end_dim=-1)  # (B, C, L)
            x = x.permute(0, 2, 1)  # convert to channel-last, (B, L, C)
        for idx, module in enumerate(self):
            x = module(x)
            if idx == 3:  # before the last linear layer
                x = torch.mean(x, dim=1)  # do global average pooling on the length dimension
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=768):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
            ),
            num_layers=4,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
    
    def forward(self, x, return_cls_token=False):
        # concat the [CLS] token to the input
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.encoder(x)
        if return_cls_token:
            return self.classifier(x[:, 0]), x[:, 0]
        else:
            return self.classifier(x[:, 0])


class FairMedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        num_sensitive_classes=2,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.linear_layer = nn.Linear(768, num_sensitive_classes)
        self.vpt_linear_layer = nn.Linear(768, num_sensitive_classes)
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # # sensitive attribute predictor
        # self.sensitive_predictor = Predictor(num_classes=num_sensitive_classes)

    def forward(self, image, box, training_mode=True):
        # generate embeddings from encoder
        # result = self.image_encoder(image)  # 打印实际返回值
        # print(len(result))
        # print("image_encoder output:", type(result), result)

        # feature_map, image_tokens, visual_prompts, _ = self.image_encoder(image)  # (B, 256, 64, 64), (B, N, 768), (B, num_tokens, 768)
        feature_map, image_tokens, visual_prompts = self.image_encoder(image)
        # print("feature_map:", feature_map.shape)
        # print("image_tokens:", image_tokens.shape)
        # print("visual_prompts:", visual_prompts.shape)
        
        # 1. run the segmentation
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=feature_map,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # # 2. run the sensitive attribute prediction
        # img_pred = self.sensitive_predictor(image_tokens)
        # vpt_pred = self.sensitive_predictor(visual_prompts)

        # return ori_res_masks, img_pred, vpt_pred if training_mode else ori_res_masks
        if training_mode:
            return ori_res_masks, image_tokens, visual_prompts
        return ori_res_masks


class ShannonEntropy():
    def __init__(self, use_sigmoid=False, use_softmax=False, reduction="mean"):
        self.use_sigmoid = use_sigmoid
        self.use_softmax = use_softmax
        self.reduction = reduction
        self.eps = 1e-6

    def __call__(self, x):
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        if self.use_softmax:
            x = F.softmax(x, dim=1)
        e = -torch.sum(x * torch.log(x + self.eps), dim=1)
        if self.reduction == "mean":
            return e.mean()
        elif self.reduction == "sum":
            return e.sum()
        else:
            return e
