"""
Aerial Vehicle Detector  ->  INT8 PTQ  ->  FPGA (archway_npu) export
=====================================================================
Target board : Arty A7-100T running Linux, archway_npu systolic-array NPU.

This mirrors the proven railway-classifier flow (train.py) but for a
YOLO-style *object detector* on the merged aerial-vehicle dataset
(classes: 0=small-vehicle, 1=large-vehicle, 2=human).

WHAT RUNS WHERE  (identical split to the classifier that already works):
  * HARDWARE (systolic array): every 3x3 convolution  (conv only).
  * HOST CPU (in new_infer.c): requant/ReLU/maxpool are folded into the
    accelerator's post_proc; the CPU does the input quantize, the 1x1
    detection HEAD (float), box decode and NMS.

HARDWARE TRUTHS taken straight from the RTL (src/*.v), which differ from a
"normal" accelerator and MUST be matched by training or the board diverges:
  1. post_proc.v  : NO BIAS.  out = clip((acc*qscale) >> qshift, 0,255).
                    => the conv backbone is trained bias-free; folded BN bias
                       is dropped on HW (biases.bin kept only for the file
                       contract, never added on the FPGA path).
  2. pe.v line 29 : pixel_in is $signed -> activations are read back as
                    SIGNED int8.  A post-ReLU value of 200 becomes -56 next
                    layer.  => we keep activations in [0,127] (signed==unsigned
                    there) and we MODEL the wrap exactly in the int8 sim, so
                    the reported INT8 mAP equals what the board produces.
  3. weight BRAM  : 512 rows  -> Cin*K*K <= 512  (3x3 => Cin <= 56, 1x1 => Cin <= 512).
  4. NUM_PE = 128 -> Cout <= 128 per layer.  cfg_k is 2 bits -> K in {1,2,3}.
  5. act bank     : two 32768-byte banks (ping-pong). BUT the HOST can only address
                    14 bits (16384) per bank over AXI-Lite -- bit14 is stolen for
                    bank-select in cnn_top.v / archway_npu_v1_0_S00_AXI.v -- and
                    new_infer.c reads EVERY layer back through the host.  So every
                    layer's input AND output tensor must be <= 16384 bytes, or its
                    upper half is read back as garbage.  (The old 32x32x32=32768
                    stem violated this: it only "worked" for the classifier because
                    global-average-pool hides the corruption; a detector cannot.)

DETECTOR (anchor-free, single scale, tiled) -- deeper, all tensors <= 16384:
  Backbone (all on NPU), 64x64x3 tile in:
     L0  3x3 s2  Cout=16          64->32   out 32x32x16 = 16384   (Cin*9=27)
     L1  3x3 s2  Cout=48          32->16   out 16x16x48 = 12288   (Cin*9=144)
     L2  3x3 s1  Cout=56          16->16   out 16x16x56 = 14336   (Cin*9=432)
     L3  3x3 s1  Cout=56          16->16   out 16x16x56 = 14336   (Cin*9=504)
     L4  3x3 s1  Cout=56          16->16   out 16x16x56 = 14336   (Cin*9=504)
     L5  3x3 s1  Cout=56          16->16   out 16x16x56 = 14336   (Cin*9=504)
     L6  1x1 s1  Cout=64          16->16   out 16x16x64 = 16384   (Cin*1=56)
     L7  1x1 s1  Cout=64          16->16   out 16x16x64 = 16384   (Cin*1=64)
  -> feature map 64 x 16 x 16  (stride 4 inside the tile).
  HEAD (CPU, float 1x1 conv): 64 -> (5 + n_classes) per cell.
     channels = [obj, tx, ty, tw, th, cls0..cls_{C-1}]
  Decode (per cell gx,gy, stride S=4):
     cx=(gx+sigmoid(tx))*S  cy=(gy+sigmoid(ty))*S
     bw=exp(tw)*S           bh=exp(th)*S
     score = sigmoid(obj) * softmax(cls).max()

TILING:  the full image is squashed to CANVAS x CANVAS, split into 64x64
tiles; the detector runs per tile and detections are offset to canvas
coords, then global per-class NMS.  Training uses random 64x64 crops of the
CANVAS-resized image so train/inference object scale matches.

VAL:  the dataset's val/ folder was deleted, so we carve a deterministic
val split out of train/ (default 4000 images) for early-stopping, mAP and
PTQ calibration.

Outputs (model/):
  weights.bin     int8 conv weights, OIHW, concatenated per layer
  biases.bin      int32 folded biases per layer (file contract; HW ignores)
  model_arch.h    layer table + quant scales + float detection head + decode
  model_int8.pt   quantized state (record)
  test_bins/      a few CANVAS x CANVAS x3 uint8 .bin test images + GT
"""

import os, sys, glob, random, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG (override via CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    data_root = "/home/joeld/aerial-dataset/RoadVehiclesYOLODatasetPro",
    out_dir   = "model",
    calib_dir = "calib",
    canvas    = 384,     # full image squashed to canvas x canvas (multiple of tile; 384 -> 6x6=36 tiles)
    tile      = 64,      # NPU tile size (the 64x64 the HW is balanced for)
    epochs    = 60,
    patience  = 12,
    batch     = 64,
    lr        = 2e-3,
    seed      = 42,
    n_val     = 4000,    # images held out of train/ for validation
    n_calib   = 200,     # images for PTQ activation calibration
    calib_pct = 100.0,   # activation calibration percentile.  100 = map each layer's
                         #   MAX activation to calib_denom, so NOTHING lands in 128..255
                         #   (which the board reinterprets as SIGNED negative = the wrap).
                         #   This is the DEPLOYED setting: it took the board from 0.0065
                         #   (99.9 = a wrapping tail) to 0.1194 == the no-wrap ceiling, no
                         #   retrain.  Lower it only if a freak calib outlier over-compresses
                         #   (watch the INT8-nowrap diag line); pair with calib_denom headroom.
    calib_denom = 127.0, # the calib percentile is mapped to THIS value (not 255).  LOWER
                         #   (e.g. 100) buys headroom below 128 so the >127 wrap is rarely
                         #   hit on HW, at the cost of activation precision.  Sweep this on
                         #   the bias-free model with NO retrain and watch the INT8 line.
    n_eval    = 400,     # val images used for the FINAL mAP report (speed); 0 = all
    crops_per_img = 2,   # random 64x64 crops drawn per image per epoch (denser
                         #   positive supervision; epoch is crops_per_img x longer)
    map_every = 2,       # run a (fast) FP32 mAP eval every N epochs for checkpoint
                         #   selection -- we keep the best-mAP model, NOT best-loss
                         #   (the old run early-stopped on val_loss and kept mAP~0)
    map_eval_n= 150,     # val images used for the in-loop mAP selection (cheap)
    qat_epochs= 0,       # 0 = pure PTQ (DEFAULT -- run this FIRST). >0 = WHOLE-NETWORK
                         #     QAT: finetune the whole net through the bias-free int8
                         #     datapath, warm-started from FP32, selecting on INT8 val mAP.
                         #     WARNING: the last --qat_epochs 30 run DIVERGED (train_loss
                         #     5.2e6) and never beat INT8 0.0000 -- do NOT re-run QAT until
                         #     the diagnostic ladder (printed by pure PTQ) confirms it is
                         #     the right fix and the divergence is understood.
    qat_lr    = 2e-4,    # QAT finetune LR (warmup then cosine to 0). Kept LOW: starting
                         #     from the bias-free-broken point, 1e-3 detonated the loss
                         #     (head logits ran to +-800 in ep1, loss 4.7e6 in ep2).
    biasfree_epochs = 0, # >0 = FLOAT finetune of the WHOLE net through the deployed
                         #     bias-free forward (drop the BN bias the board cannot apply).
                         #     This attacks the real wall: BiasFree-float=0.0 means the net
                         #     dies the instant the bias is dropped, EVEN in float, before
                         #     any int8.  Pure float (no int8 STE) so it is stable, unlike
                         #     QAT.  Warm-start from model_fp32.pt: --biasfree_epochs 25.
    biasfree_lr = 1e-3,  # LR for the bias-free finetune (cosine to 0). Float path is
                         #     stable; grad-clip 5.0 guards the initial bias-drop shock.
    resume_fp32 = "",    # path to a saved FP32 checkpoint (model_fp32.pt). If set, SKIP
                         #     training and go straight to PTQ+eval+export -- re-quantize
                         #     in ~1 min instead of retraining for hours.
    obj_alpha = 0.25,    # focal-loss alpha for objectness (RetinaNet fg/bg balance).
    obj_gamma = 2.0,     #   focal gamma. Replaces the old pos_weight=32 BCE that made
                         #   the net fire objectness EVERYWHERE (flooded 3500 boxes/img).
    cls_balance = True,  # weight class CE by inverse sqrt-frequency (dataset is ~320:99:7
                         #   small:human:large -> head was collapsing to small-vehicle)
    lambda_box= 5.0,
    lambda_cls= 1.0,
    conf_thr  = 0.25,    # deployment score threshold baked into the header (board)
    eval_conf = 0.01,    # LOW threshold used ONLY for mAP (mAP must sweep the full
                         #   PR curve; using the deployment 0.25 here truncates recall
                         #   and crushes mAP to ~0 even for a good model)
    eval_topk = 100,     # max cells kept per tile during mAP eval
    eval_max_det = 300,  # GLOBAL cap on dets/image before NMS (bounds the O(n^2) NMS;
                         #   without it a flooding model makes eval take hours)
    nms_iou   = 0.45,
    workers   = 8,
)
CLASS_NAMES = ["small-vehicle", "large-vehicle", "human"]
NUM_CLASSES = len(CLASS_NAMES)

# Backbone spec: (out_ch, kernel, stride, pool_after).  pad = k//2.
# Sized so EVERY layer's input AND output tensor is <= 16384 bytes (the host can
# only reach 14 bits per activation bank -- see HARDWARE TRUTH #5).  3x3 layers
# obey Cin*9 <= 512 (=> Cin <= 56); 1x1 layers widen channels cheaply (Cin <= 512).
# Two stride-2 layers take 64 -> 16 (grid 16, stride 4); no maxpool needed.
HW_LAYERS = [
    (16, 3, 2, False),   # L0 stem  64->32   32x32x16 = 16384  (Cin*9=27)
    (48, 3, 2, False),   # L1       32->16   16x16x48 = 12288  (Cin*9=144)
    (56, 3, 1, False),   # L2       16->16   16x16x56 = 14336  (Cin*9=432)
    (56, 3, 1, False),   # L3       16->16   16x16x56 = 14336  (Cin*9=504)
    (56, 3, 1, False),   # L4       16->16   16x16x56 = 14336  (Cin*9=504)
    (56, 3, 1, False),   # L5       16->16   16x16x56 = 14336  (Cin*9=504)
    (64, 1, 1, False),   # L6 1x1   widen    16x16x64 = 16384  (Cin*1=56)
    (64, 1, 1, False),   # L7 1x1   mix      16x16x64 = 16384  (Cin*1=64) -> feature
]
ACT_HOST_LIMIT = 16384   # bytes the host can address per activation bank (bit14=bank)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def tensor_budget(spec, tile, limit=ACT_HOST_LIMIT):
    """Simulate the spatial pyramid for a TILE x TILE input and return per-layer
    (idx, cin, cout, h, w, oh, ow, in_elems, out_elems, ok).  ok=False means a
    tensor exceeds what the host can read/write per bank -> would corrupt on board."""
    h = w = tile; c = 3; rows = []
    for i, (cout, k, s, pool) in enumerate(spec):
        pad = k // 2
        oh = (h + 2 * pad - k) // s + 1
        ow = (w + 2 * pad - k) // s + 1
        in_e, out_e = c * h * w, cout * oh * ow
        rows.append((i, c, cout, h, w, oh, ow, in_e, out_e, in_e <= limit and out_e <= limit))
        h, w, c = oh, ow, cout
        if pool:
            h //= 2; w //= 2
    return rows


# ===========================================================================
# DATA  (YOLO format: <split>/images/*.jpg  + <split>/labels/*.txt
#        label line: cls cx cy w h   (all normalized 0..1))
# ===========================================================================
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(images_dir):
    files = []
    for e in IMG_EXT:
        files += glob.glob(os.path.join(images_dir, "*" + e))
        files += glob.glob(os.path.join(images_dir, "*" + e.upper()))
    return sorted(set(files))


def label_path_for(img_path):
    d = os.path.dirname(os.path.dirname(img_path))          # .../<split>
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(d, "labels", base + ".txt")


def read_label(img_path):
    """Return list of (cls, cx, cy, w, h) normalized; [] if none/missing."""
    lp = label_path_for(img_path)
    boxes = []
    if not os.path.isfile(lp):
        return boxes
    with open(lp, "r") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
            if w <= 0 or h <= 0 or c < 0 or c >= NUM_CLASSES:
                continue
            boxes.append((c, cx, cy, w, h))
    return boxes


def count_classes(files, n_sample=4000):
    """Count GT instances per class over a sample of files (to derive CE weights)."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    rng = random.Random(0)
    sample = files if (n_sample <= 0 or len(files) <= n_sample) else rng.sample(files, n_sample)
    for p in sample:
        for (c, _, _, _, _) in read_label(p):
            counts[c] += 1
    return counts


class AerialCropDataset(Dataset):
    """Training set: resize image to CANVAS, take a random TILE x TILE crop and
    build the per-cell detection target grid for that crop. With prob 0.5 the
    crop is centered on a random GT box (so the net sees positives often)."""

    def __init__(self, img_files, canvas, tile, stride, grid, train=True, seed=0,
                 crops_per_img=1):
        self.files = img_files
        self.canvas = canvas
        self.tile = tile
        self.stride = stride
        self.grid = grid            # cells per tile side (tile // stride)
        self.train = train
        self.rng = random.Random(seed)
        self.mult = max(1, int(crops_per_img))

    def __len__(self):
        return len(self.files) * self.mult

    def _load_canvas_boxes(self, path):
        img = Image.open(path).convert("RGB").resize((self.canvas, self.canvas))
        arr = np.asarray(img, dtype=np.uint8)               # HWC
        gts = []
        for (c, cx, cy, w, h) in read_label(path):
            gts.append((c, cx * self.canvas, cy * self.canvas,
                        w * self.canvas, h * self.canvas))   # canvas px, center form
        return arr, gts

    def _pick_crop_origin(self, gts):
        T, C = self.tile, self.canvas
        maxo = C - T
        if maxo <= 0:
            return 0, 0
        if self.train and gts and self.rng.random() < 0.5:
            _, gx, gy, _, _ = gts[self.rng.randrange(len(gts))]
            ox = int(gx - T / 2 + self.rng.randint(-T // 3, T // 3))
            oy = int(gy - T / 2 + self.rng.randint(-T // 3, T // 3))
        else:
            ox = self.rng.randint(0, maxo)
            oy = self.rng.randint(0, maxo)
        ox = max(0, min(maxo, ox))
        oy = max(0, min(maxo, oy))
        return ox, oy

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        try:
            arr, gts = self._load_canvas_boxes(path)
        except Exception:
            arr = np.zeros((self.canvas, self.canvas, 3), np.uint8); gts = []
        ox, oy = self._pick_crop_origin(gts)
        T = self.tile
        crop = arr[oy:oy + T, ox:ox + T, :]                 # HWC uint8

        # quantize to [0,127] (signed-safe for the NPU's signed pixel read)
        x = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,T,T] in [0,1]

        G, S = self.grid, self.stride
        tobj = torch.zeros(G, G)
        tbox = torch.zeros(4, G, G)        # tx,ty (offset 0..1), tw,th (log size/stride)
        tcls = torch.full((G, G), -1, dtype=torch.long)
        for (c, gx, gy, bw, bh) in gts:
            lx, ly = gx - ox, gy - oy
            if lx < 0 or ly < 0 or lx >= T or ly >= T:
                continue                                    # center outside crop
            bw = max(1.0, min(bw, T)); bh = max(1.0, min(bh, T))
            cellx = min(G - 1, int(lx // S)); celly = min(G - 1, int(ly // S))
            tobj[celly, cellx] = 1.0
            tbox[0, celly, cellx] = (lx / S) - cellx        # in (0,1)
            tbox[1, celly, cellx] = (ly / S) - celly
            tbox[2, celly, cellx] = math.log(bw / S + 1e-6)
            tbox[3, celly, cellx] = math.log(bh / S + 1e-6)
            tcls[celly, cellx] = c
        return x, tobj, tbox, tcls


# ===========================================================================
# MODEL
# ===========================================================================
class ConvBNReLU(nn.Module):
    """Conv(bias=False) + BN + ReLU.  bias=False because the NPU has no bias;
    BN gives stable FP32 training and is folded away before quantization."""
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class DetNet(nn.Module):
    def __init__(self, spec=HW_LAYERS, n_classes=NUM_CLASSES):
        super().__init__()
        self.spec = spec
        self.na = 5 + n_classes
        c0 = 3
        blocks = []
        for (cout, k, s, _p) in spec:
            blocks.append(ConvBNReLU(c0, cout, k, s)); c0 = cout
        self.blocks = nn.ModuleList(blocks)
        self.pools = [p for (_c, _k, _s, p) in spec]
        self.feat_ch = c0
        self.head = nn.Conv2d(c0, self.na, kernel_size=1, bias=True)  # 1x1 (per-cell FC)
        # RetinaNet objectness prior: start predicting p(obj)~0.01 everywhere so the
        # net does NOT flood objectness at init (focal loss then stays stable). This
        # was a big contributor to the old "100k detections / mAP~0" failure.
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = -4.595      # logit for sigmoid()=0.01
        self._check_limits()

    def _check_limits(self):
        c = 3
        for i, (cout, k, s, _p) in enumerate(self.spec):
            assert cout <= 128, f"L{i}: Cout {cout} > 128 (NUM_PE)"
            assert c * k * k <= 512, f"L{i}: Cin*K*K {c*k*k} > 512 (weight BRAM)"
            c = cout

    def backbone(self, x):
        for blk, pool in zip(self.blocks, self.pools):
            x = blk(x)
            if pool:
                x = F.max_pool2d(x, 2)
        return x                                            # [B, feat_ch, G, G]

    def forward(self, x):
        return self.head(self.backbone(x))                  # [B, na, G, G]


# ===========================================================================
# DETECTION LOSS
# ===========================================================================
def bbox_ciou(p, g, eps=1e-7):
    """Complete-IoU between predicted and target boxes given in center form
    (cx, cy, w, h) on the LAST axis.  Returns CIoU (~[-1,1]); box loss = 1 - CIoU.
    CIoU optimises the exact overlap mAP scores -- the decoupled xy/wh MSE the old
    loss used drove val_loss down while boxes still missed at IoU>=0.5."""
    pcx, pcy, pw, ph = p.unbind(-1)
    gcx, gcy, gw, gh = g.unbind(-1)
    px1, py1, px2, py2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    gx1, gy1, gx2, gy2 = gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2
    iw = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
    ih = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter + eps
    iou = inter / union
    cw = torch.max(px2, gx2) - torch.min(px1, gx1)
    ch = torch.max(py2, gy2) - torch.min(py1, gy1)
    c2 = cw * cw + ch * ch + eps
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    v = (4.0 / (math.pi ** 2)) * (torch.atan(gw / (gh + eps)) -
                                  torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return iou - rho2 / c2 - alpha * v


def det_loss(pred, tobj, tbox, tcls, cfg):
    """pred [B,na,G,G]; targets as built by the dataset.
    Objectness uses FOCAL loss (alpha,gamma) normalized by #positives -- this is
    the RetinaNet fix for the extreme background/foreground imbalance, and it stops
    the net from firing objectness everywhere (the old pos_weight=32 BCE flooded the
    image with ~3500 boxes). Class CE is inverse-frequency weighted (cfg['class_w'])
    so the head stops collapsing to the dominant small-vehicle class."""
    B, na, G, _ = pred.shape
    obj_l = pred[:, 0]                                      # [B,G,G]
    tx, ty, tw, th = pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    cls_l = pred[:, 5:]                                     # [B,C,G,G]

    pos = tobj > 0.5
    npos = pos.sum().clamp(min=1).float()

    # ---- focal objectness over ALL cells, normalized by #positives ----
    alpha, gamma = cfg["obj_alpha"], cfg["obj_gamma"]
    p = torch.sigmoid(obj_l)
    ce = F.binary_cross_entropy_with_logits(obj_l, tobj, reduction="none")
    p_t = p * tobj + (1.0 - p) * (1.0 - tobj)
    alpha_t = alpha * tobj + (1.0 - alpha) * (1.0 - tobj)
    obj_loss = (alpha_t * (1.0 - p_t).pow(gamma) * ce).sum() / npos

    if pos.any():
        # --- CIoU box loss: decode pred AND target boxes to tile pixels, 1 - CIoU.
        #     Uses the EXACT decode new_infer.c applies, so we optimise the overlap
        #     metric the board's boxes will be scored on.
        S = float(cfg["stride"])
        dev = pred.device
        gx = torch.arange(G, device=dev).view(1, 1, G).float()   # column index (x)
        gy = torch.arange(G, device=dev).view(1, G, 1).float()   # row index    (y)
        pcx = (gx + torch.sigmoid(tx)) * S
        pcy = (gy + torch.sigmoid(ty)) * S
        pw  = torch.exp(torch.clamp(tw, -4.0, 4.0)) * S    # two-sided: bounds the exp
        ph  = torch.exp(torch.clamp(th, -4.0, 4.0)) * S    #   gradient (exp(4)*S=216px >> tile)
        gcx = (gx + tbox[:, 0]) * S
        gcy = (gy + tbox[:, 1]) * S
        gw  = torch.exp(tbox[:, 2]) * S
        gh  = torch.exp(tbox[:, 3]) * S
        p_box = torch.stack([pcx, pcy, pw, ph], dim=-1)          # [B,G,G,4]
        g_box = torch.stack([gcx, gcy, gw, gh], dim=-1)
        ciou = bbox_ciou(p_box, g_box)                           # [B,G,G]
        box_loss = (1.0 - ciou)[pos].mean()
        # class CE on positive cells (inverse-frequency weighted)
        cls_pred = cls_l.permute(0, 2, 3, 1)[pos]                # [Npos, C]
        cls_tgt = tcls[pos]
        cls_loss = F.cross_entropy(cls_pred, cls_tgt, weight=cfg.get("class_w"))
    else:
        box_loss = pred.sum() * 0.0
        cls_loss = pred.sum() * 0.0

    total = obj_loss + cfg["lambda_box"] * box_loss + cfg["lambda_cls"] * cls_loss
    return total, obj_loss.detach(), box_loss.detach(), cls_loss.detach()


# ===========================================================================
# BN FOLDING -> bias-free conv weights (HW has no bias; we keep b only for
# the biases.bin file contract and for the optional software reference)
# ===========================================================================
def fold_bn(conv, bn):
    w = conv.weight.detach().clone()
    gamma, beta = bn.weight.detach(), bn.bias.detach()
    mean, var, eps = bn.running_mean.detach(), bn.running_var.detach(), bn.eps
    std = torch.sqrt(var + eps)
    w_f = w * (gamma / std).reshape(-1, 1, 1, 1)
    b_f = beta - mean * gamma / std                         # conv had bias=False
    return w_f, b_f


def collect_folded_layers(model):
    layers = []
    for m, (cout, k, s, pool) in zip(model.blocks, model.spec):
        w, b = fold_bn(m.conv, m.bn)
        layers.append(dict(weight=w, bias=b,
                           in_ch=m.conv.in_channels, out_ch=m.conv.out_channels,
                           k=m.conv.kernel_size[0], stride=m.conv.stride[0],
                           pad=m.conv.padding[0], pool=bool(pool)))
    return layers


# ===========================================================================
# PTQ  (symmetric weights /127 ; activations unsigned mapped to [0,127] ;
#       requant qscale/qshift exactly as post_proc.v does it)
# ===========================================================================
def gather_calib_tensor(files, canvas, tile, grid, stride, dev, n):
    """Build a calibration batch of TILE crops (centered on objects when present)."""
    ds = AerialCropDataset(files[:max(n, 1)], canvas, tile, stride, grid, train=True, seed=123)
    xs = []
    for i in range(min(n, len(ds))):
        x, *_ = ds[i]
        xs.append(x)
    return torch.stack(xs).to(dev)


def quantize_export(model, calib_x, cfg, dev):
    grid, stride = cfg["grid"], cfg["stride"]
    layers = collect_folded_layers(model)
    head_w = model.head.weight.detach().reshape(model.na, model.feat_ch).clone()  # [na, Cf]
    head_b = model.head.bias.detach().clone()

    # ---- calibration: run the FOLDED, BIAS-FREE float backbone and record the
    #      conv INPUT (signed, [-1,1]-ish) and post-ReLU OUTPUT activations. ----
    ws = [l["weight"].to(dev) for l in layers]
    acts_in, acts_out = [[] for _ in layers], [[] for _ in layers]
    with torch.no_grad():
        x = calib_x
        for i, l in enumerate(layers):
            acts_in[i].append(x.detach())
            y = F.conv2d(x, ws[i], None, stride=l["stride"], padding=l["pad"])  # NO bias (HW truth)
            y = F.relu(y)
            acts_out[i].append(y.detach())
            x = y
            if l["pool"]:
                x = F.max_pool2d(x, 2)

    def pos_scale(t, denom):                               # nonneg activations -> [0,denom]
        a = t.flatten()
        k = max(1, min(a.numel(), int(round(a.numel() * cfg["calib_pct"] / 100.0))))
        v = torch.kthvalue(a, k).values.item()
        return max(v / denom, 1e-8)

    # input scale: image is in [0,1]; map 1.0 -> 127 so q in [0,127]
    in_scales, out_scales = [], []
    for i, l in enumerate(layers):
        if i == 0:
            in_scales.append(1.0 / 127.0)
        else:
            in_scales.append(out_scales[i - 1])
        out_scales.append(pos_scale(torch.cat(acts_out[i]), cfg["calib_denom"]))

    export = []
    for i, l in enumerate(layers):
        w = l["weight"]; b = l["bias"]
        w_absmax = w.abs().max().item()
        sw = (w_absmax / 127.0) if w_absmax > 0 else 1e-8
        w_q = torch.clamp(torch.round(w / sw), -127, 127).to(torch.int8)
        sx = in_scales[i]
        s_acc = sx * sw
        b_q = torch.round(b / s_acc).to(torch.int32)        # file contract only
        s_out = out_scales[i]
        M = s_acc / s_out
        qshift = 0
        for sh in range(31, -1, -1):
            if round(M * (1 << sh)) <= 65535:
                qshift = sh; break
        qscale = max(1, min(65535, int(round(M * (1 << qshift)))))
        export.append(dict(**l, w_q=w_q, b_q=b_q, sx=sx, sw=sw, s_acc=s_acc,
                           s_out=s_out, qscale=qscale, qshift=qshift))
    last_s_out = export[-1]["s_out"]
    return export, head_w, head_b, last_s_out


# ===========================================================================
# HARDWARE-EXACT INT8 SIMULATOR (matches new_infer.c and the board byte-for-byte)
#   - conv input read as SIGNED int8 (values >=128 -> v-256)
#   - acc integer, out = clip((acc*qscale)>>qshift, 0,255), ReLU
#   - maxpool on UNSIGNED [0,255]
#   - final feature dequantized UNSIGNED * last_s_out, then float head
# ===========================================================================
def quant_input(x01):
    """[0,1] float image -> integer-valued float in [0,127] (input quant)."""
    return torch.clamp(torch.round(x01 / (1.0 / 127.0)), 0, 127)


def biasfree_float_features(export, x01):
    """DIAGNOSTIC: folded FLOAT backbone (no quant) with the BN bias DROPPED --
    the exact function the bias-free FPGA must approximate.  Decides the fork:
      ~FP32 here => the net works bias-free; only activation quant is left ->
                    a no-retrain fix (cross-layer equalization) can deploy.
      ~0   here => the net NEEDS the BN bias the HW cannot apply -> QAT (train a
                    bias-free, quant-robust model) is mandatory."""
    x = x01
    for e in export:
        w = e["weight"].to(x.device)                        # full-precision folded weight, NO bias
        x = F.relu(F.conv2d(x, w, None, stride=e["stride"], padding=e["pad"]))
        if e["pool"]:
            x = F.max_pool2d(x, 2)
    return x


def wq_only_features(export, x01):
    """DIAGNOSTIC (HW-impossible): weights quantized to int8 (per-tensor,
    dequantized w_q*sw), activations kept in FLOAT, BN bias DROPPED (bias-free,
    matching the deployed reality), float ReLU, NO requant/clamp/wrap, float
    [0,1] input.  Isolates how much per-tensor WEIGHT quant alone costs vs the
    activation int8 path, IN THE BIAS-FREE REGIME.
      ~BiasFree-float here => weights quantize for free; the killer is the int8
                              ACTIVATION path (rounding / requant / >127 wrap).
      ~0.00 here           => per-tensor weight quant itself is fatal -> QAT."""
    x = x01                                                 # [B,3,T,T] float [0,1], real units
    for e in export:
        w = e["w_q"].float().to(x.device) * e["sw"]         # dequantized int8 weights, NO bias
        x = F.relu(F.conv2d(x, w, None, stride=e["stride"], padding=e["pad"]))
        if e["pool"]:
            x = F.max_pool2d(x, 2)
    return x                                                # float feature for the head


def int8_features(export, x_uint_127, add_bias=False, clamp_max=255):
    """Exact integer datapath -> dequantized last feature map [B,Cf,G,G].
    x_uint_127: [B,3,T,T] float holding integers in [0,127] (input quant).

    DIAGNOSTIC flags (default off = byte-exact FPGA; the board can do NEITHER):
      add_bias : add the folded-BN bias back in the integer domain (b_q) before
                 requant -- isolates how much the HW bias-drop costs.  NOT
                 deployable: post_proc.v / new_infer.c never add bias.
      clamp_max: 255 = exact HW (post_proc clips to uint8, then the next conv
                 reads it SIGNED so 128..255 wrap to negative).  127 = no-wrap
                 (clip below the sign bit) -- isolates how much that wrap costs."""
    xq = x_uint_127
    last_uint = xq
    for e in export:
        w_q = e["w_q"].float().to(xq.device)
        acc = F.conv2d(xq, w_q, None, stride=e["stride"], padding=e["pad"])
        if add_bias:                                        # diagnostic only (HW has no bias adder)
            acc = acc + e["b_q"].float().to(acc.device).view(1, -1, 1, 1)
        out = torch.floor(acc * e["qscale"] / float(1 << e["qshift"]))
        out = torch.clamp(out, 0, clamp_max)                # uint8, ReLU (clamp_max=255 = exact HW)
        if e["pool"]:
            out = F.max_pool2d(out, 2)                      # unsigned max
        # signed reinterpret for the NEXT conv (pe.v reads $signed)
        xq = torch.where(out >= 128, out - 256.0, out)
        last_uint = out                                     # UNSIGNED for the head
    return last_uint * export[-1]["s_out"]                  # dequantized feature


# ===========================================================================
# DECODE + NMS  (shared by FP32 eval, INT8 eval and matches new_infer.c)
# ===========================================================================
def decode_tile(pred, stride, conf_thr):
    """pred [na,G,G] torch -> list of (x1,y1,x2,y2,score,cls) in TILE coords."""
    na, G, _ = pred.shape
    obj = torch.sigmoid(pred[0])
    cls = torch.softmax(pred[5:], dim=0)
    cls_conf, cls_id = cls.max(0)
    score = obj * cls_conf
    ys, xs = torch.where(score > conf_thr)
    out = []
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        tx = torch.sigmoid(pred[1, gy, gx]).item()
        ty = torch.sigmoid(pred[2, gy, gx]).item()
        tw = pred[3, gy, gx].item(); th = pred[4, gy, gx].item()
        cx = (gx + tx) * stride; cy = (gy + ty) * stride
        bw = math.exp(min(tw, 6.0)) * stride; bh = math.exp(min(th, 6.0)) * stride
        out.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2,
                    score[gy, gx].item(), int(cls_id[gy, gx].item())])
    return out


def decode_grid_np(pred, stride, conf_thr, topk):
    """Fast, numerically-identical vectorized version of decode_tile for mAP eval.
    pred: [na,G,G] numpy (already on CPU). Caps to top-K cells so NMS stays bounded
    (at eval_conf=0.01 thousands of background cells pass -> per-cell .item() on GPU
    tensors + O(n^2) Python NMS would hang)."""
    obj = 1.0 / (1.0 + np.exp(-pred[0]))
    cl = pred[5:] - pred[5:].max(axis=0, keepdims=True)
    e = np.exp(cl); sm = e / e.sum(axis=0, keepdims=True)
    cls_id = sm.argmax(axis=0); cls_conf = sm.max(axis=0)
    score = obj * cls_conf
    ys, xs = np.where(score > conf_thr)
    if ys.size == 0:
        return []
    sc = score[ys, xs]
    if topk > 0 and sc.size > topk:
        idx = np.argpartition(-sc, topk)[:topk]
        ys, xs, sc = ys[idx], xs[idx], sc[idx]
    tx = 1.0 / (1.0 + np.exp(-pred[1, ys, xs]))
    ty = 1.0 / (1.0 + np.exp(-pred[2, ys, xs]))
    tw = np.minimum(pred[3, ys, xs], 6.0); th = np.minimum(pred[4, ys, xs], 6.0)
    cx = (xs + tx) * stride; cy = (ys + ty) * stride
    bw = np.exp(tw) * stride; bh = np.exp(th) * stride
    cid = cls_id[ys, xs]
    return [[float(cx[i] - bw[i] / 2), float(cy[i] - bh[i] / 2),
             float(cx[i] + bw[i] / 2), float(cy[i] + bh[i] / 2),
             float(sc[i]), int(cid[i])] for i in range(sc.size)]


def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, iou_thr):
    """dets: list [x1,y1,x2,y2,score,cls]; per-class greedy NMS."""
    keep = []
    for c in set(d[5] for d in dets):
        cd = sorted([d for d in dets if d[5] == c], key=lambda d: -d[4])
        while cd:
            best = cd.pop(0); keep.append(best)
            cd = [d for d in cd if iou_xyxy(best, d) < iou_thr]
    return keep


def infer_full(per_tile_fn, canvas, tile, stride, conf_thr, nms_iou):
    """per_tile_fn(ox,oy)->pred[na,G,G]; tiles canvas, returns canvas-coord dets."""
    dets = []
    nt = canvas // tile
    for ty in range(nt):
        for tx in range(nt):
            ox, oy = tx * tile, ty * tile
            pred = per_tile_fn(ox, oy)
            for d in decode_tile(pred, stride, conf_thr):
                dets.append([d[0] + ox, d[1] + oy, d[2] + ox, d[3] + oy, d[4], d[5]])
    return nms(dets, nms_iou)


# ===========================================================================
# mAP@0.5  (per-class AP, then mean)
# ===========================================================================
def compute_map(all_dets, all_gts, n_classes, iou_thr=0.5):
    details = []   # (class, ap, npos, ndet)
    for c in range(n_classes):
        # gather detections of class c across images, sorted by score desc
        D = []
        npos = 0
        gt_used = {}
        for img_id, gts in all_gts.items():
            cg = [g for g in gts if g[4] == c]
            gt_used[img_id] = [False] * len(cg)
            npos += len(cg)
        for img_id, dets in all_dets.items():
            for d in dets:
                if d[5] == c:
                    D.append((d[4], img_id, d[:4]))
        D.sort(key=lambda t: -t[0])
        if npos == 0:
            continue
        tp = np.zeros(len(D)); fp = np.zeros(len(D))
        for i, (score, img_id, box) in enumerate(D):
            cg = [g for g in all_gts[img_id] if g[4] == c]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(cg):
                v = iou_xyxy(box, g[:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and not gt_used[img_id][best_j]:
                tp[i] = 1; gt_used[img_id][best_j] = True
            else:
                fp[i] = 1
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c / (npos + 1e-9)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
            ap += p / 101.0
        details.append((c, ap, npos, len(D)))
    mean = float(np.mean([d[1] for d in details])) if details else 0.0
    return mean, details


def gts_for_image(path, canvas):
    out = []
    for (c, cx, cy, w, h) in read_label(path):
        x, y, bw, bh = cx * canvas, cy * canvas, w * canvas, h * canvas
        out.append([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, c])
    return out


def load_canvas_input(path, canvas, dev):
    img = Image.open(path).convert("RGB").resize((canvas, canvas))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).to(dev)    # [3,canvas,canvas]


def evaluate_map(model, export, files, cfg, dev, use_int8, add_bias=False, clamp_max=255,
                 wq_only=False, biasfree_float=False):
    """Run full tiling inference over `files` and compute mAP@0.5.
    All tiles of an image run in ONE batched forward and decode is vectorized on
    the CPU (decode_grid_np) -- decoding at eval_conf=0.01 cell-by-cell on GPU
    tensors would hang. Numerically identical to decode_tile (the C reference)."""
    model.eval()
    canvas, tile, stride = cfg["canvas"], cfg["tile"], cfg["stride"]
    conf, topk, nms_iou = cfg["eval_conf"], cfg["eval_topk"], cfg["nms_iou"]
    max_det = cfg["eval_max_det"]
    nt = canvas // tile
    all_dets, all_gts = {}, {}
    with torch.no_grad():
        for path in files:
            full = load_canvas_input(path, canvas, dev)          # [3,C,C] [0,1]
            tiles = torch.stack([full[:, ty * tile:ty * tile + tile,
                                          tx * tile:tx * tile + tile]
                                 for ty in range(nt) for tx in range(nt)], 0)  # [B,3,T,T]
            if biasfree_float:
                feat = biasfree_float_features(export, tiles)  # float backbone, NO bias (diag)
                preds = model.head(feat)
            elif wq_only:
                feat = wq_only_features(export, tiles)        # weights int8, acts float (diag)
                preds = model.head(feat)
            elif use_int8:
                feat = int8_features(export, quant_input(tiles),
                                     add_bias=add_bias, clamp_max=clamp_max)  # [B,Cf,G,G] dequant
                preds = model.head(feat)
            else:
                preds = model(tiles)
            preds = preds.cpu().numpy()                          # [B,na,G,G]

            dets, b = [], 0
            for ty in range(nt):
                for tx in range(nt):
                    ox, oy = tx * tile, ty * tile
                    for d in decode_grid_np(preds[b], stride, conf, topk):
                        dets.append([d[0] + ox, d[1] + oy, d[2] + ox, d[3] + oy, d[4], d[5]])
                    b += 1
            dets.sort(key=lambda d: -d[4])                    # global top-K before NMS
            all_dets[path] = nms(dets[:max_det], nms_iou)
            all_gts[path] = gts_for_image(path, canvas)
    return compute_map(all_dets, all_gts, NUM_CLASSES)


# ===========================================================================
# TRAIN
# ===========================================================================
def run_epoch(model, loader, cfg, dev, opt=None, export=None):
    train = opt is not None
    model.train(train)
    tot = tob = tbo = tcl = n = 0.0
    for x, tobj, tbox, tcls in loader:
        x, tobj, tbox, tcls = x.to(dev), tobj.to(dev), tbox.to(dev), tcls.to(dev)
        if export is not None:                               # QAT: refine head on frozen int8 features
            feat = int8_features(export, quant_input(x))     # no grad to backbone (fixed int weights)
            pred = model.head(feat)
        else:
            pred = model(x)
        loss, ob, bo, cl = det_loss(pred, tobj, tbox, tcls, cfg)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        bs = x.size(0); n += bs
        tot += loss.item() * bs; tob += ob.item() * bs
        tbo += float(bo) * bs; tcl += float(cl) * bs
    return tot / n, tob / n, tbo / n, tcl / n


# ===========================================================================
# QAT  (whole-network finetune through the EXACT bias-free int8 datapath)
#   The board collapses for TWO independent, each-fatal reasons (proven by the
#   diagnostic ladder): it cannot add the folded-BN bias, and per-tensor int8
#   activation quant over 8 layers washes the signal out.  Neither is fixable
#   post-hoc.  So we finetune the WHOLE net (conv weights + BN gamma + head)
#   with the bias-free integer datapath simulated in the forward via straight-
#   through estimators.  Activations are held to the [0,127] (no-wrap) grid, so
#   this fake-quant forward is byte-identical to int8_features there -- and we
#   SELECT the checkpoint on the real INT8 val mAP, not a proxy.
# ===========================================================================
def _fq_weight(w):
    """Per-tensor symmetric int8 fake-quant of a weight tensor (STE)."""
    s = (w.detach().abs().max() / 127.0).clamp(min=1e-8)
    wq = torch.clamp(torch.round(w / s), -127, 127) * s
    return w + (wq - w).detach()                            # straight-through gradient


def _fq_act(y, s):
    """Unsigned activation fake-quant to the [0,127] grid = ReLU+clamp (STE).
    Clamping at 127 (not 255) keeps the net in the no-wrap region where the sim
    equals the FPGA exactly."""
    q = torch.clamp(torch.round(y / s), 0.0, 127.0) * s
    return y + (q - y).detach()                             # straight-through gradient


def _fold_bn_diff(conv, bn):
    """Differentiable BN fold -> bias-free weight (HW drops the bias).  Frozen
    running stats; conv.weight and gamma carry gradient, beta is dropped."""
    std = torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)


def qat_forward(model, x01, scales):
    """Differentiable replica of int8_features: bias-free, per-tensor int8
    weights, [0,127] activations, then the float head.  scales[i] = per-layer
    activation LSB from calibration."""
    x = _fq_act(x01, 1.0 / 127.0)                           # input quant [0,1]->[0,127]
    for i, blk in enumerate(model.blocks):
        w = _fq_weight(_fold_bn_diff(blk.conv, blk.bn))
        y = F.relu(F.conv2d(x, w, None, stride=blk.conv.stride[0], padding=blk.conv.padding[0]))
        if model.pools[i]:
            y = F.max_pool2d(y, 2)
        x = _fq_act(y, scales[i])
    return model.head(x)


def qat_finetune(model, tr_dl, map_files, calib_x, cfg, dev):
    """Warm-started whole-network QAT, HARDENED against divergence (the first
    attempt at lr=1e-3 detonated: loss 5 -> 4.7e6 in two epochs).  Defences:
    LOW lr (cfg['qat_lr']) with a linear warmup then cosine decay, tight grad
    clip (1.0), a non-finite-step guard that drops blown-up batches, and
    activation scales refreshed only every 5 epochs (re-deriving them from
    degrading weights every epoch fed the death spiral).  Keeps the checkpoint
    with the best EXACT INT8 val mAP, printed live."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["qat_lr"], weight_decay=1e-4)
    steps_per_epoch = max(1, len(tr_dl))
    total_steps = cfg["qat_epochs"] * steps_per_epoch
    warmup = min(300, max(1, total_steps // 20))
    base_lr = cfg["qat_lr"]
    best_map, best_state, best_ep = -1.0, None, -1
    scales, gstep = None, 0
    for ep in range(cfg["qat_epochs"]):
        if scales is None or ep % 5 == 0:                  # refresh scales sparingly
            export, _, _, _ = quantize_export(model, calib_x, cfg, dev)
            scales = [e["s_out"] for e in export]
        model.train()
        tot = n = nskip = 0.0
        for x, tobj, tbox, tcls in tr_dl:
            lr = (base_lr * (gstep + 1) / warmup if gstep < warmup else
                  0.5 * base_lr * (1.0 + math.cos(math.pi * (gstep - warmup) /
                                                  max(1, total_steps - warmup))))
            for g in opt.param_groups:
                g["lr"] = lr
            gstep += 1
            x, tobj, tbox, tcls = x.to(dev), tobj.to(dev), tbox.to(dev), tcls.to(dev)
            pred = qat_forward(model, x, scales)
            loss, _, _, _ = det_loss(pred, tobj, tbox, tcls, cfg)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):                   # guard: drop a blown-up batch
                nskip += 1; continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):                  # guard: drop non-finite grads
                opt.zero_grad(set_to_none=True); nskip += 1; continue
            opt.step()
            bs = x.size(0); n += bs; tot += loss.item() * bs
        export, _, _, _ = quantize_export(model, calib_x, cfg, dev)   # EXACT int8 eval
        m, _ = evaluate_map(model, export, map_files, cfg, dev, use_int8=True)
        flag = ""
        if m > best_map:
            best_map, best_ep = m, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <-- best"
        print(f"[qat {ep+1:02d}] train_loss={tot/max(n,1):.3f} skip={int(nskip)} "
              f"lr={lr:.2e} | INT8_val_mAP={m:.4f} (best {best_map:.4f} @ep{best_ep+1}){flag}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] restored best-INT8 weights (INT8_val_mAP={best_map:.4f} @ep{best_ep+1})")
    return model


def train_fp32(model, tr_dl, map_files, cfg, dev):
    """Train FP32, selecting the checkpoint by val mAP@0.5 (NOT val loss).  The old
    run early-stopped on val_loss and kept a model with mAP~0 -- loss and mAP were
    decoupled.  We now keep the highest-mAP weights and early-stop on mAP plateau."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    best_map, best_state, best_ep = -1.0, None, -1
    for ep in range(cfg["epochs"]):
        tr = run_epoch(model, tr_dl, cfg, dev, opt=opt)
        sched.step()
        line = (f"[ep {ep+1:02d}] train_loss={tr[0]:.3f} "
                f"(obj {tr[1]:.3f} box {tr[2]:.3f} cls {tr[3]:.3f})")
        do_eval = ((ep + 1) % cfg["map_every"] == 0) or (ep + 1 == cfg["epochs"])
        if do_eval:
            m, _ = evaluate_map(model, None, map_files, cfg, dev, use_int8=False)
            flag = ""
            if m > best_map:
                best_map, best_ep = m, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                flag = "  <-- best"
            print(f"{line} | val_mAP={m:.4f} (best {best_map:.4f} @ep{best_ep+1}){flag}")
            if (ep - best_ep) >= cfg["patience"]:
                print(f"[INFO] early stop at epoch {ep+1}, best val_mAP={best_map:.4f}")
                break
        else:
            print(line)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] restored best-mAP weights (val_mAP={best_map:.4f} @ep{best_ep+1})")
    return model


def biasfree_float_forward(model, x01):
    """The DEPLOYED function minus int8 quant: the bias-free folded backbone (drops
    beta AND the BN mean term -- exactly what new_infer.c does, the board has no bias
    adder) with FLOAT activations, then the float head.  Finetuning THIS optimises
    BiasFree-float, which is 0.0 for a BN-trained net = the real wall.  Grads flow to
    conv.weight and the BN gamma via _fold_bn_diff; beta / running_mean are dropped."""
    x = x01
    for i, blk in enumerate(model.blocks):
        w = _fold_bn_diff(blk.conv, blk.bn)                 # bias-free folded weight
        x = F.relu(F.conv2d(x, w, None, stride=blk.conv.stride[0], padding=blk.conv.padding[0]))
        if model.pools[i]:
            x = F.max_pool2d(x, 2)
    return model.head(x)


def eval_biasfree_map(model, files, cfg, dev):
    """mAP of the bias-free FLOAT function (matches biasfree_float_forward) -- no
    quantize_export needed, just the folded full-precision weights."""
    layers = collect_folded_layers(model)
    return evaluate_map(model, layers, files, cfg, dev, use_int8=False, biasfree_float=True)


def biasfree_finetune(model, tr_dl, map_files, cfg, dev):
    """Stable FLOAT finetune of the WHOLE net through the bias-free forward, so the
    net learns to work WITHOUT the BN bias the board drops.  No int8 STE here (that is
    what detonated QAT: loss 5 -> 5e6) -- this is ordinary float training, as stable as
    train_fp32.  Selects the checkpoint on the bias-free-float val mAP."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["biasfree_lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["biasfree_epochs"])
    best_map, best_state, best_ep = -1.0, None, -1
    for ep in range(cfg["biasfree_epochs"]):
        model.train()
        tot = n = 0.0
        for x, tobj, tbox, tcls in tr_dl:
            x, tobj, tbox, tcls = x.to(dev), tobj.to(dev), tbox.to(dev), tcls.to(dev)
            pred = biasfree_float_forward(model, x)
            loss, _, _, _ = det_loss(pred, tobj, tbox, tcls, cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = x.size(0); n += bs; tot += loss.item() * bs
        sched.step()
        line = f"[bf {ep+1:02d}] train_loss={tot/max(n,1):.3f}"
        if ((ep + 1) % cfg["map_every"] == 0) or (ep + 1 == cfg["biasfree_epochs"]):
            m, _ = eval_biasfree_map(model, map_files, cfg, dev)
            flag = ""
            if m > best_map:
                best_map, best_ep = m, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                flag = "  <-- best"
            print(f"{line} | biasfree_val_mAP={m:.4f} (best {best_map:.4f} @ep{best_ep+1}){flag}")
        else:
            print(line)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] restored best bias-free weights (biasfree_val_mAP={best_map:.4f} @ep{best_ep+1})")
    return model


# ===========================================================================
# EXPORT FILES
# ===========================================================================
def write_files(export, head_w, head_b, last_s_out, cfg):
    out_dir = cfg["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    wpath = os.path.join(out_dir, "weights.bin")
    bpath = os.path.join(out_dir, "biases.bin")
    hpath = os.path.join(out_dir, "model_arch.h")

    woff = boff = 0; rows = []
    with open(wpath, "wb") as fw, open(bpath, "wb") as fb:
        for e in export:
            wb = e["w_q"].cpu().numpy().tobytes()            # int8 OIHW
            bb = e["b_q"].cpu().numpy().tobytes()            # int32
            fw.write(wb); fb.write(bb)
            rows.append((e, woff, len(wb), boff, len(bb)))
            woff += len(wb); boff += len(bb)

    hw = head_w.cpu().numpy().astype(np.float32)             # [na, Cf]
    hb = head_b.cpu().numpy().astype(np.float32)             # [na]
    na = hw.shape[0]; cf = hw.shape[1]

    with open(hpath, "w") as h:
        h.write("#ifndef MODEL_ARCH_H\n#define MODEL_ARCH_H\n#include <stdint.h>\n\n")
        h.write("/* Auto-generated by new_train.py. Matches archway_npu RTL:\n"
                "   - conv on the systolic array, NO bias (post_proc.v)\n"
                "   - activations UNSIGNED uint8 written back, read $signed (pe.v)\n"
                "     => we keep them in [0,127]; requant = clip((acc*qscale)>>qshift,0,255)\n"
                "   - per layer: Cout<=128, Cin*K*K<=512, tensor<=32768 bytes\n"
                "   - HEAD (1x1) + decode + NMS run on the host CPU (float). */\n\n")
        h.write(f"#define CANVAS        {cfg['canvas']}\n")
        h.write(f"#define TILE          {cfg['tile']}\n")
        h.write(f"#define N_TILES       {cfg['canvas'] // cfg['tile']}\n")
        h.write(f"#define INPUT_CH      3\n")
        h.write(f"#define GRID          {cfg['grid']}      /* cells per tile side */\n")
        h.write(f"#define STRIDE        {cfg['stride']}      /* tile px per cell  */\n")
        h.write(f"#define NUM_CLASSES   {NUM_CLASSES}\n")
        h.write(f"#define NA            {na}      /* head outputs/cell = 5 + NUM_CLASSES */\n")
        h.write(f"#define LAST_CONV_CH  {cf}\n")
        h.write(f"static const float INPUT_SCALE   = {1.0/127.0:.10e}f;"
                f"  /* q = clip(round((px/255)/INPUT_SCALE),0,127) */\n")
        h.write(f"static const float LAST_ACC_SCALE = {last_s_out:.10e}f;"
                f"  /* feat = uint8_out * LAST_ACC_SCALE */\n")
        h.write(f"static const float CONF_THRESH   = {cfg['conf_thr']:.4f}f;\n")
        h.write(f"static const float NMS_IOU       = {cfg['nms_iou']:.4f}f;\n")
        names = ", ".join(f'"{n}"' for n in CLASS_NAMES)
        h.write(f"static const char* CLASS_NAMES[NUM_CLASSES] = {{{names}}};\n\n")
        h.write("typedef struct {\n"
                "    uint32_t in_ch, out_ch;\n"
                "    uint32_t k_h, k_w;\n"
                "    uint32_t stride, pad;\n"
                "    uint32_t pool;                       /* 1 = 2x2 maxpool after conv */\n"
                "    uint32_t relu;\n"
                "    uint32_t weight_offset, weight_size; /* bytes into weights.bin (int8) */\n"
                "    uint32_t bias_offset,   bias_size;   /* bytes into biases.bin (int32, HW unused) */\n"
                "    uint32_t qscale;  uint32_t qshift;\n"
                "} layer_config_t;\n\n")
        h.write(f"#define NUM_LAYERS {len(rows)}\n\n")
        h.write("static const layer_config_t model_layers[NUM_LAYERS] = {\n")
        for (e, wo, ws, bo, bs) in rows:
            h.write(f"    {{{e['in_ch']}, {e['out_ch']}, {e['k']}, {e['k']}, "
                    f"{e['stride']}, {e['pad']}, {1 if e['pool'] else 0}, 1, "
                    f"{wo}, {ws}, {bo}, {bs}, {e['qscale']}, {e['qshift']}}},\n")
        h.write("};\n\n")
        # detection head: DET_W[NA][LAST_CONV_CH], DET_B[NA]
        h.write("static const float DET_W[NA][LAST_CONV_CH] = {\n")
        for o in range(na):
            vals = ", ".join(f"{v:.8e}f" for v in hw[o])
            h.write(f"    {{{vals}}},\n")
        h.write("};\n")
        h.write("static const float DET_B[NA] = {" + ", ".join(f"{v:.8e}f" for v in hb) + "};\n\n")
        h.write("#endif /* MODEL_ARCH_H */\n")
    print(f"[INFO] wrote {wpath} ({woff} B), {bpath} ({boff} B), {hpath}")
    return rows


def export_test_bins(files, cfg, n=4):
    out = os.path.join(cfg["out_dir"], "test_bins"); os.makedirs(out, exist_ok=True)
    rng = random.Random(cfg["seed"] + 7)
    picked = files[:]; rng.shuffle(picked); picked = picked[:n]
    for k, path in enumerate(picked):
        img = Image.open(path).convert("RGB").resize((cfg["canvas"], cfg["canvas"]))
        np.asarray(img, np.uint8).tofile(os.path.join(out, f"sample_{k}.bin"))
        with open(os.path.join(out, f"sample_{k}.gt.txt"), "w") as f:
            for g in gts_for_image(path, cfg["canvas"]):
                f.write(f"{int(g[4])} {g[0]:.1f} {g[1]:.1f} {g[2]:.1f} {g[3]:.1f}\n")
    print(f"[INFO] wrote {len(picked)} test .bin ({cfg['canvas']}x{cfg['canvas']}x3 uint8 HWC) -> {out}/")


def hw_check(export, cfg):
    print("\n[HW-CHECK] every layer vs the accelerator RTL "
          f"(host bank limit = {ACT_HOST_LIMIT} B/tensor):")
    h = w = cfg["tile"]; c = 3; ok_all = True
    if c * h * w > ACT_HOST_LIMIT:
        print(f"  input {c}x{h}x{w}={c*h*w} > {ACT_HOST_LIMIT} FAIL"); ok_all = False
    for i, e in enumerate(export):
        k, s, pad = e["k"], e["stride"], e["pad"]
        oh = (h + 2 * pad - k) // s + 1; ow = (w + 2 * pad - k) // s + 1
        in_e = e["in_ch"] * h * w; out_e = e["out_ch"] * oh * ow
        mac = e["in_ch"] * k * k
        c1 = e["out_ch"] <= 128; c2 = mac <= 512
        c3 = out_e <= ACT_HOST_LIMIT and in_e <= ACT_HOST_LIMIT   # host-readable both ways
        c4 = e["qscale"] <= 65535 and e["qshift"] <= 31
        c5 = max(oh, ow) <= 127
        ok = c1 and c2 and c3 and c4 and c5; ok_all &= ok
        print(f"  L{i}: Cin={e['in_ch']:3d} Cout={e['out_ch']:3d} {h}x{w}->{oh}x{ow} "
              f"mac={mac:3d} q={e['qscale']:5d}>>{e['qshift']:2d} "
              f"in={in_e:5d} out={out_e:5d} {'OK' if ok else 'FAIL'}")
        h, w, c = oh, ow, e["out_ch"]
        if e["pool"]:
            h //= 2; w //= 2
    print(f"  ==> {'ALL CONSTRAINTS SATISFIED' if ok_all else 'CONSTRAINT VIOLATION'}")
    return ok_all


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    cfg = vars(ap.parse_args())
    assert cfg["canvas"] % cfg["tile"] == 0, "canvas must be a multiple of tile"
    set_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # derive grid/stride from the backbone (downsample = product of strides*pools)
    model = DetNet().to(dev)
    with torch.no_grad():
        probe = model.backbone(torch.zeros(1, 3, cfg["tile"], cfg["tile"], device=dev))
    cfg["grid"] = probe.shape[-1]
    cfg["stride"] = cfg["tile"] // cfg["grid"]
    print(f"[INFO] device={dev} canvas={cfg['canvas']} tile={cfg['tile']} "
          f"grid={cfg['grid']} stride={cfg['stride']} feat_ch={model.feat_ch}")

    # fail fast: every layer's in/out tensor must fit the host's per-bank window
    budget = tensor_budget(HW_LAYERS, cfg["tile"])
    bad = [r for r in budget if not r[-1]]
    if bad:
        for (i, cin, cout, h, w, oh, ow, in_e, out_e, _) in bad:
            print(f"[FATAL] L{i} {cin}x{h}x{w}->{cout}x{oh}x{ow}: "
                  f"in={in_e} out={out_e} exceeds host bank limit {ACT_HOST_LIMIT}")
        sys.exit("[FATAL] backbone exceeds the 16384 B/bank host limit -> would corrupt on board")

    # --- data split (val/ was deleted -> carve from train/) ---
    train_imgs_dir = os.path.join(cfg["data_root"], "train", "images")
    files = list_images(train_imgs_dir)
    if not files:
        sys.exit(f"[FATAL] no images under {train_imgs_dir}")
    rng = random.Random(cfg["seed"]); rng.shuffle(files)
    n_val = min(cfg["n_val"], len(files) // 5)
    val_files, train_files = files[:n_val], files[n_val:]
    print(f"[INFO] train={len(train_files)} val={len(val_files)} (carved from train/)")

    # class imbalance -> inverse-sqrt-frequency CE weights (normalized ~1)
    counts = count_classes(train_files, 4000)
    if cfg["cls_balance"] and counts.sum() > 0:
        freq = counts / max(counts.sum(), 1)
        w = 1.0 / np.sqrt(freq + 1e-6)
        w = w / w.mean()
        cfg["class_w"] = torch.tensor(w, dtype=torch.float32, device=dev)
    else:
        cfg["class_w"] = None
    print(f"[INFO] class counts {dict(zip(CLASS_NAMES, counts.tolist()))}  "
          f"weights={None if cfg['class_w'] is None else [round(v,2) for v in cfg['class_w'].tolist()]}")

    tr_ds = AerialCropDataset(train_files, cfg["canvas"], cfg["tile"], cfg["stride"], cfg["grid"],
                              True, cfg["seed"], crops_per_img=cfg["crops_per_img"])
    tr_dl = DataLoader(tr_ds, cfg["batch"], shuffle=True, num_workers=cfg["workers"],
                       drop_last=True, pin_memory=(dev == "cuda"))
    map_files = val_files[:cfg["map_eval_n"]]   # small subset for in-loop mAP selection

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] DetNet params={n_params:,}  layers={len(model.spec)}  "
          f"head={model.na}x{model.feat_ch}  crops/img={cfg['crops_per_img']}")

    # --- FP32 training (checkpoint selected by val mAP, not loss) ---
    fp32_ckpt = os.path.join(cfg["out_dir"], "model_fp32.pt")
    if cfg["resume_fp32"]:
        print(f"[INFO] resume_fp32: loading FP32 weights from {cfg['resume_fp32']} (SKIP training)")
        model.load_state_dict(torch.load(cfg["resume_fp32"], map_location=dev))
        m0, _ = evaluate_map(model, None, map_files, cfg, dev, use_int8=False)
        print(f"[INFO] loaded FP32 model val_mAP={m0:.4f} (sanity check)")
    else:
        model = train_fp32(model, tr_dl, map_files, cfg, dev)
        torch.save(model.state_dict(), fp32_ckpt)
        print(f"[INFO] saved FP32 weights -> {fp32_ckpt}")
        print(f"[INFO]   re-quantize without retraining:  "
              f"python new_train.py --resume_fp32 {fp32_ckpt}")

    # --- BIAS-FREE FLOAT finetune: teach the net to work WITHOUT the BN bias the
    #     board drops (BiasFree-float was 0.0 = the real wall; the board literally
    #     cannot add bias -- new_infer.c:26).  Stable float training (no int8 STE),
    #     warm-started from the FP32 weights.  Produces model_biasfree_fp32.pt. ---
    if cfg["biasfree_epochs"] > 0:
        print(f"[INFO] bias-free FLOAT finetune for {cfg['biasfree_epochs']} epochs "
              f"(lr={cfg['biasfree_lr']}) -- optimises the deployed bias-free function")
        model = biasfree_finetune(model, tr_dl, map_files, cfg, dev)
        bf_ckpt = os.path.join(cfg["out_dir"], "model_biasfree_fp32.pt")
        torch.save(model.state_dict(), bf_ckpt)
        print(f"[INFO] saved bias-free model -> {bf_ckpt}")
        print(f"[INFO]   re-quantize without retraining:  python new_train.py --resume_fp32 {bf_ckpt}")

    # --- PTQ on the (bias-free, if finetuned) FP32 model ---
    calib_x = gather_calib_tensor(train_files, cfg["canvas"], cfg["tile"], cfg["grid"], cfg["stride"], dev, cfg["n_calib"])
    export, head_w, head_b, last_s_out = quantize_export(model, calib_x, cfg, dev)
    hw_check(export, cfg)

    eval_files = val_files if cfg["n_eval"] == 0 else val_files[:cfg["n_eval"]]
    print(f"[INFO] evaluating mAP@0.5 on {len(eval_files)} val images "
          f"(decode thr={cfg['eval_conf']}) ...")

    def show(tag, mean, details):
        print(f"\n[{tag} mAP@0.5] {mean:.4f}")
        for (c, ap_c, npos, ndet) in details:
            print(f"    {CLASS_NAMES[c]:<14s} AP={ap_c:.4f}  (gt={npos}, det={ndet})")

    # --- mAP (FP32) + DIAGNOSTIC ladder on the PRISTINE model.  These MUST run
    #     BEFORE any QAT: qat_finetune() overwrites `model` in place, so the old
    #     order evaluated the QAT-wrecked net and printed it as "FP32" -- that is
    #     why the last log showed FP32 0.0000 even though the loaded model was
    #     0.1484 (the sanity-check line proves the weights were fine).
    map_fp, det_fp = evaluate_map(model, export, eval_files, cfg, dev, use_int8=False)
    map_q,  det_q  = evaluate_map(model, export, eval_files, cfg, dev, use_int8=True)
    show("FP32", map_fp, det_fp)
    show("INT8 (pure PTQ)", map_q,  det_q)
    print("   (INT8 is exactly what the FPGA produces)")

    # --- DIAGNOSTIC ladder: isolate WHY int8 collapses (neither is deployable;
    #     the FPGA has no bias adder and no >127 guard -- see new_infer.c:27).
    #     Read it as: how much mAP each pathology costs, to pick the real fix.
    #       INT8                 = bias dropped + wrap        (the board)
    #       INT8+bias            = bias restored + wrap       (isolates bias-drop)
    #       INT8+bias+nowrap     = bias restored + clamp<=127 (isolates the wrap)
    # BIAS-FREE decomposition (all NO-bias, matching the deployed board).  Reads as a
    # ladder from the float deployed-math down to the exact board, so the 0.13->0.006
    # int8 collapse is split into its independent causes:
    #   BiasFree-float : float weights, float acts, no bias  (deployed math in FLOAT)
    #   W8-Afloat      : int8 weights, float acts, no bias    (cost of WEIGHT quant)
    #   INT8-nowrap    : full int8, no bias, clamp<=127        (int8 acts, WRAP removed)
    #   INT8 (above)   : full int8, no bias, clamp<=255 + wrap (the board)
    map_bf, det_bf = evaluate_map(model, export, eval_files, cfg, dev,
                                  use_int8=False, biasfree_float=True)
    map_w,  det_w  = evaluate_map(model, export, eval_files, cfg, dev,
                                  use_int8=False, wq_only=True)
    map_nw, det_nw = evaluate_map(model, export, eval_files, cfg, dev,
                                  use_int8=True, add_bias=False, clamp_max=127)
    show("BiasFree-float  (DIAG: float backbone, NO bias = deployed math in FLOAT)", map_bf, det_bf)
    show("W8-Afloat  (DIAG: int8 WEIGHTS, float acts, NO bias -- cost of weight quant)", map_w, det_w)
    show("INT8-nowrap  (DIAG: full int8, NO bias, clamp<=127 -- the >127 wrap removed)", map_nw, det_nw)
    print("   read the bias-free ladder  BiasFree-float -> W8-Afloat -> INT8-nowrap -> INT8:")
    print("     big drop at W8-Afloat      => WEIGHT quant is fatal       -> QAT")
    print("     big drop at INT8-nowrap    => activation rounding/requant -> activation QAT")
    print("     big drop INT8-nowrap->INT8 => the >127 SIGNED WRAP -> raise --calib_pct (99.99) or")
    print("        lower --calib_denom (e.g. 100) for wrap headroom, NO retrain")

    # --- OPTIONAL QAT (DEFAULT OFF).  Warm-starts from the pristine net and
    #     finetunes the WHOLE net through the bias-free int8 datapath.  It modifies
    #     `model` IN PLACE, which is why it now runs AFTER the honest report above.
    #     NOTE: the last attempt diverged (train_loss -> 5.2e6) and never beat
    #     INT8 0.0000 over 30 epochs -- read the diagnostic ladder above FIRST to
    #     confirm QAT is even the right fix before paying for it again.
    if cfg["qat_epochs"] > 0:
        print(f"\n[INFO] QAT: whole-net finetune for {cfg['qat_epochs']} epochs through the "
              f"bias-free int8 datapath (lr={cfg['qat_lr']}, selects on INT8 val mAP)")
        model = qat_finetune(model, tr_dl, map_files, calib_x, cfg, dev)
        export, head_w, head_b, last_s_out = quantize_export(model, calib_x, cfg, dev)
        hw_check(export, cfg)
        qat_ckpt = os.path.join(cfg["out_dir"], "model_qat_fp32.pt")
        torch.save(model.state_dict(), qat_ckpt)
        print(f"[INFO] saved QAT-finetuned weights -> {qat_ckpt}")
        map_qat, det_qat = evaluate_map(model, export, eval_files, cfg, dev, use_int8=True)
        show("INT8 (after QAT -- DEPLOYED)", map_qat, det_qat)
        print("   (this -- not the pure-PTQ line above -- is what the FPGA runs when QAT is used)")

    # --- write export files (QAT export if QAT ran, else the pristine PTQ) ---
    write_files(export, head_w, head_b, last_s_out, cfg)
    torch.save({"export": [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in e.items()} for e in export],
                "head_w": head_w.cpu(), "head_b": head_b.cpu(),
                "cfg": {k: cfg[k] for k in ("canvas", "tile", "grid", "stride", "conf_thr", "nms_iou")}},
               os.path.join(cfg["out_dir"], "model_int8.pt"))
    export_test_bins(val_files, cfg)
    print(f"\n[DONE] -> '{cfg['out_dir']}/': weights.bin biases.bin model_arch.h model_int8.pt test_bins/")


if __name__ == "__main__":
    main()
