import mmcv
import torch
import torch.nn.functional as F
from mmseg.models import EncoderDecoder
from utils import get_logger


class DINOTextSegInference(EncoderDecoder):
    def __init__(
            self,
            model,
            text_embedding,
            classnames,
            with_bg,
            test_cfg=dict(),
            pamr=False,
            bg_thresh=0.5,
            bg_strategy="base",
            **kwargs,
    ):
        super(EncoderDecoder, self).__init__()

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.bg_strategy = bg_strategy

        self.model = model
        self.register_buffer("text_embedding", text_embedding)
        self.classnames = classnames
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(text_embedding) + 1
        else:
            self.num_classes = len(text_embedding)
        self.out_channels = self.num_classes
        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building DINOTextSegInference with {self.num_classes} classes, test_cfg={test_cfg}, with_bg={with_bg}"
            f", pamr={pamr}, bg_thresh={bg_thresh}"
        )

    def encode_decode(self, img, img_metas):
        assert img.shape[0] == 1, "batch size must be 1"
        masks, simmap = self.model.generate_masks(
            img,
            img_metas,
            self.text_embedding,
            self.classnames,
            apply_pamr=self.pamr,
        )

        B, N, H, W = masks.shape

        if self.with_bg:
            masks = masks.cpu()
            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
            masks = masks.to(img.device)

        return masks


class OracleCropAugSegInference(EncoderDecoder):
    """
    Same inference protocol as DINOTextSegInference, except that the class
    prototypes are NOT text embeddings. They are visual prototypes built from
    train cropaug_patch_tokens + GT part masks.
    """
    def __init__(
            self,
            model,
            visual_prototypes,
            classnames,
            with_bg,
            test_cfg=dict(),
            pamr=False,
            bg_thresh=0.5,
            bg_strategy="base",
            **kwargs,
    ):
        super(EncoderDecoder, self).__init__()

        if not isinstance(test_cfg, mmcv.Config):
            test_cfg = mmcv.Config(test_cfg)
        self.test_cfg = test_cfg
        self.pamr = pamr
        self.bg_thresh = bg_thresh
        self.bg_strategy = bg_strategy

        self.model = model
        self.register_buffer("visual_prototypes", visual_prototypes)
        self.classnames = classnames
        self.with_bg = with_bg
        if self.with_bg:
            self.num_classes = len(visual_prototypes) + 1
        else:
            self.num_classes = len(visual_prototypes)
        self.out_channels = self.num_classes
        self.align_corners = False
        logger = get_logger()
        logger.info(
            f"Building OracleCropAugSegInference with {self.num_classes} classes, test_cfg={test_cfg}, "
            f"with_bg={with_bg}, pamr={pamr}, bg_thresh={bg_thresh}"
        )

    def encode_decode(self, img, img_metas):
        assert img.shape[0] == 1, "batch size must be 1"
        masks, simmap = self.model.generate_masks(
            img,
            img_metas,
            self.visual_prototypes,
            self.classnames,
            apply_pamr=self.pamr,
        )

        B, N, H, W = masks.shape

        if self.with_bg:
            masks = masks.cpu()
            background = torch.full(
                [B, 1, H, W], self.bg_thresh, dtype=torch.float, device=masks.device
            )
            masks = torch.cat([background, masks], dim=1)
            masks = masks.to(img.device)

        return masks
