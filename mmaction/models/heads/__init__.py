from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .fbo_head import FBOHead
from .i3d_head import I3DHead
from .lfb_infer_head import LFBInferHead
from .roi_head import AVARoIHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .tpn_head import TPNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .fc_head import FCHead
from .r50_head import R50Head, R50Head2
from .r50_multi_head import R50MultiHead
from .timesformer_head import TimeSformerHead
__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'BBoxHeadAVA', 'AVARoIHead',
    'FBOHead', 'LFBInferHead', 'FCHead', 'R50Head', 'R50Head2', 'R50MultiHead', 'TimeSformerHead'
]
