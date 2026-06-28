"""
Railway Defect Binary Classifier  ->  INT8 PTQ  ->  FPGA export
================================================================
Pipeline:
  1. Train a compact, *pure-CNN* VGG-style classifier in FP32 (transfer-friendly,
     but strictly conv -> BN -> ReLU -> maxpool, so it maps to a systolic array).
  2. Early stopping with patience on validation accuracy.
  3. Fold BatchNorm into the preceding Conv (hardware never sees BN).
  4. Post-Training Quantization (PTQ) to INT8 using a calibration set.
  5. Export:
        model/weights.bin    (int8 conv weights, concatenated, per-layer)
        model/biases.bin     (int32 conv biases,  concatenated, per-layer)
        model/model_arch.h    (layer table + quant scales + FC head)
        model/model_int8.pt   (the quantized model state, for record)
        calib/                (copied calibration images)

Layer split for "max CPU offload":
  - HARDWARE (systolic array): the 3x3 convolutions only.
  - HOST CPU (in the .c file): bias add, requantize, ReLU, 2x2 maxpool,
    global-average-pool, and the final fully-connected (FC) classifier.
"""

import os, shutil, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------------------------------------------------------
# CONFIG  (override via CLI flags)
# --------------------------------------------------------------------------
DEFAULTS = dict(
    data_root = "RailwayDefectDataset",   # contains train/ val/ test/
    out_dir   = "model",
    calib_dir = "calib",
    img_size  = 64,                        # 64x64 RGB  (matches your FPGA)
    epochs    = 120,
    patience  = 20,
    batch     = 16,
    lr        = 1e-3,
    seed      = 42,
    n_calib   = 64,                        # images sampled for PTQ calibration
    defect_weight = 1.5,                    # CE weight on the defective class (recall lever)
)
CLASSES = ["defective", "non-defective"]   # index 0 / 1  -> fixed, do not reorder

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# --------------------------------------------------------------------------
# MODEL : hardware-legal pure CNN  (one conv per stage; obeys every RTL limit)
#
#   HARD LIMITS taken from the Verilog:
#     * NUM_PE = 128            -> C_out <= 128 per layer
#     * weight BRAM 512 rows    -> Cin*K*K <= 512  (3x3 => Cin <= 56)
#     * act bank = 32768 int8   -> input AND output tensor each <= 32768
#     * feature dims 7-bit      -> H,W <= 127 (we stay far below)
#     * post_proc ReLU clamp    -> activations are UNSIGNED uint8 [0,255]
#
#   To fit 64x64x3 without tiling: the FIRST conv uses stride 2, so we never
#   materialize a 64x64xC map (that would be 64*64*16 = 65536 > 32768).
#
#   Per-stage config: (out_ch, kernel, stride, pool_after)
#     L0: 16, 3, s2, no-pool   64->32   out 16x32x32=16384  Cin=3   ok
#     L1: 32, 3, s1, pool      32->32->16  out 32x32x32=32768 Cin=16  ok
#     L2: 48, 3, s1, pool      16->16->8   out 48x16x16=12288 Cin=32  ok
#     L3: 56, 3, s1, pool       8->8->4    out 56x8x8=3584    Cin=48  ok
#     L4: 56, 3, s1, pool       4->4->2    out 56x4x4=896     Cin=56  ok (504<=512)
#   then GAP -> FC(2).
# --------------------------------------------------------------------------
HW_LAYERS = [
    # (out_ch, kernel, stride, pool_after)
    (32, 3, 2, False),   # 64->32  stem, 32 features at early resolution
    (32, 3, 1, True),    # 32->32->16
    (56, 3, 1, True),    # 16->16->8
    (56, 3, 1, True),    # 8->8->4
    (56, 3, 1, True),    # 4->4->2
]

class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=True)
        self.bn   = nn.BatchNorm2d(cout)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class RailNet(nn.Module):
    """One conv per stage; matches HW_LAYERS exactly."""
    def __init__(self, spec=HW_LAYERS, n_classes=2):
        super().__init__()
        self.spec = spec
        c0 = 3
        blocks = []
        for (cout, k, s, _pool) in spec:
            blocks.append(ConvBNReLU(c0, cout, k, s))
            c0 = cout
        self.blocks = nn.ModuleList(blocks)
        self.pools  = [p for (_c, _k, _s, p) in spec]
        self.fc = nn.Linear(c0, n_classes)
        # sanity-check the limits at construction time
        self._check_limits()
    def _check_limits(self):
        c = 3
        for i, (cout, k, s, _p) in enumerate(self.spec):
            assert cout <= 128, f"L{i}: C_out={cout} > 128 (NUM_PE)"
            assert c * k * k <= 512, f"L{i}: Cin*K*K={c*k*k} > 512 (weight BRAM)"
            c = cout
    def forward(self, x):
        for blk, pool in zip(self.blocks, self.pools):
            x = blk(x)
            if pool:
                x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # global average pool
        return self.fc(x)

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def build_loaders(cfg):
    sz = cfg["img_size"]
    # ImageFolder assigns class indices alphabetically: defective=0, non-defective=1
    train_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),                 # [0,1]
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
    ])
    root = cfg["data_root"]

    # --- resolve split folder name case-insensitively ---------------------
    # accepts Train/train, Validation/validation/val/Val, Test/test
    def find_split_dir(aliases):
        try:
            entries = {e.lower(): e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))}
        except FileNotFoundError:
            raise FileNotFoundError(f"data_root '{root}' not found. Run with "
                                    f"--data_root pointing at the folder that holds "
                                    f"the split dirs (e.g. the current directory '.').")
        for a in aliases:
            if a in entries:
                return os.path.join(root, entries[a])
        raise FileNotFoundError(f"None of {aliases} found under '{root}'. "
                                f"Present: {sorted(entries.values())}")

    # --- map any class-folder spelling onto fixed indices -----------------
    #   index 0 = defective, index 1 = non-defective.
    #   "non defective" / "non-defective" / "nondefective" -> 1 ; else if it
    #   contains "defect" (and not "non") -> 0. This keeps the label semantics
    #   correct no matter how the folders are capitalized/spaced.
    def class_index(folder_name):
        s = folder_name.lower().replace("-", " ").replace("_", " ").strip()
        is_non = s.startswith("non") or "non defect" in s
        if is_non:
            return 1
        if "defect" in s:
            return 0
        raise ValueError(f"Cannot map class folder '{folder_name}' to "
                         f"defective(0)/non-defective(1).")

    def make_dataset(split_dir, tf):
        ds = datasets.ImageFolder(split_dir, tf)
        # remap ImageFolder's alphabetical indices -> our fixed semantic indices
        remap = {old_idx: class_index(name) for name, old_idx in ds.class_to_idx.items()}
        ds.samples = [(p, remap[i]) for (p, i) in ds.samples]
        ds.targets = [remap[i] for i in ds.targets]
        ds.classes = CLASSES
        ds.class_to_idx = {CLASSES[0]: 0, CLASSES[1]: 1}
        return ds

    tr = make_dataset(find_split_dir(["train"]),              train_tf)
    va = make_dataset(find_split_dir(["val", "validation"]),  eval_tf)
    te = make_dataset(find_split_dir(["test"]),               eval_tf)
    return (DataLoader(tr, cfg["batch"], shuffle=True,  num_workers=0),
            DataLoader(va, cfg["batch"], shuffle=False, num_workers=0),
            DataLoader(te, cfg["batch"], shuffle=False, num_workers=0),
            tr, va, te)

# --------------------------------------------------------------------------
# TRAIN  (FP32) with patience
# --------------------------------------------------------------------------
def evaluate(model, loader, dev):
    model.eval(); correct = tot = 0
    tp = fp = fn = 0   # for "defective" (class 0) precision/recall
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            p = model(x).argmax(1)
            correct += (p == y).sum().item(); tot += y.numel()
            tp += ((p == 0) & (y == 0)).sum().item()
            fp += ((p == 0) & (y == 1)).sum().item()
            fn += ((p == 1) & (y == 0)).sum().item()
    acc  = correct / max(tot, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    return acc, prec, rec

def train(model, tr, va, cfg, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    # Weight the defective class (index 0) a bit higher so the model is pushed
    # to not miss defects (recall matters more than precision for a safety
    # inspection task). Tunable via --defect_weight.
    w = torch.tensor([cfg["defect_weight"], 1.0], dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    # Select the checkpoint on BALANCED accuracy = (recall_def + recall_nondef)/2,
    # which rewards catching defects instead of just overall accuracy.
    best_score, best_state, wait = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        acc, prec, rec = evaluate(model, va, dev)
        # recall of non-defective = TN/(TN+FP); approximate via the same counts
        # by re-deriving from acc/prec/rec is messy, so compute balanced acc here.
        bal = balanced_acc(model, va, dev)
        flag = ""
        if bal > best_score:
            best_score = bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0; flag = "  <-- best"
        else:
            wait += 1
        print(f"[ep {ep+1:02d}] val_acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} "
              f"bal_acc={bal:.3f} (patience {wait}/{cfg['patience']}){flag}")
        if wait >= cfg["patience"]:
            print(f"[INFO] Early stopping at epoch {ep+1}. Best bal_acc={best_score:.3f}")
            break
    model.load_state_dict(best_state)
    return model

def balanced_acc(model, loader, dev):
    """(recall_defective + recall_nondefective)/2 — robust to class imbalance."""
    model.eval()
    tp = fn = tn = fp = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            p = model(x).argmax(1)
            tp += ((p == 0) & (y == 0)).sum().item()  # defect caught
            fn += ((p == 1) & (y == 0)).sum().item()  # defect missed
            tn += ((p == 1) & (y == 1)).sum().item()  # clean correct
            fp += ((p == 0) & (y == 1)).sum().item()  # clean flagged
    rec_def = tp / max(tp + fn, 1)
    rec_cln = tn / max(tn + fp, 1)
    return 0.5 * (rec_def + rec_cln)

# --------------------------------------------------------------------------
# BN FOLDING  ->  a plain Conv (weight, bias) the hardware can run directly
# --------------------------------------------------------------------------
def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.size(0))
    gamma, beta = bn.weight.detach(), bn.bias.detach()
    mean, var, eps = bn.running_mean.detach(), bn.running_var.detach(), bn.eps
    std = torch.sqrt(var + eps)
    w_f = w * (gamma / std).reshape(-1, 1, 1, 1)
    b_f = beta + (b - mean) * gamma / std
    return w_f, b_f   # float32 fused conv

def collect_folded_layers(model):
    """Return ordered list of dicts: folded fp32 conv weight/bias + meta."""
    layers = []
    for m, (cout, k, s, pool) in zip(model.blocks, model.spec):
        w, b = fold_bn(m.conv, m.bn)
        layers.append(dict(
            weight=w, bias=b,
            in_ch=m.conv.in_channels, out_ch=m.conv.out_channels,
            k=m.conv.kernel_size[0], stride=m.conv.stride[0], pad=m.conv.padding[0],
            pool=bool(pool),
        ))
    return layers

# --------------------------------------------------------------------------
# A "folded" forward that mirrors EXACTLY what the C code will do.
#   Used both for PTQ calibration (to collect activation ranges) and to
#   verify FP32 vs INT8 agreement.
# --------------------------------------------------------------------------
class FoldedNet(nn.Module):
    """Pure conv+relu+pool+gap+fc using folded weights. No BN.
    One conv per stage; pool only where meta['pool'] is True."""
    def __init__(self, layers, fc_w, fc_b):
        super().__init__()
        self.ws = [l["weight"] for l in layers]
        self.bs = [l["bias"]   for l in layers]
        self.meta = layers
        self.fc_w, self.fc_b = fc_w, fc_b
        self.acts = []   # populated when record=True

    def forward(self, x, record=False):
        if record: self.acts = [x.detach()]
        for bi, l in enumerate(self.meta):
            x = F.conv2d(x, self.ws[bi], self.bs[bi],
                         stride=l["stride"], padding=l["pad"])
            x = F.relu(x)
            if record: self.acts.append(x.detach())
            if l["pool"]:
                x = F.max_pool2d(x, 2)
                if record: self.acts.append(x.detach())
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if record: self.acts.append(x.detach())
        x = F.linear(x, self.fc_w, self.fc_b)
        return x

# --------------------------------------------------------------------------
# PTQ  (symmetric, per-tensor, power-of-two requant shift)
#   For each conv layer i:
#       y_real = conv(x_real, w_real) + b_real
#   Quantize:
#       w_q = round(w_real / sw_i),  sw_i = max|w| / 127
#       x_q = round(x_real / sx_i),  sx_i = max|x| / 127     (input activation)
#       acc = conv(x_q, w_q)  (int32)  + b_q
#       b_q = round(b_real / (sx_i*sw_i))     -> int32, same scale as acc
#       y_real ~= acc * (sx_i * sw_i)
#   Next layer expects its input in *its* x-scale sx_{i+1}, so we requantize:
#       y_q_next = round( acc * (sx_i*sw_i) / sx_{i+1} )
#                = round( acc * M ),  M = sx_i*sw_i / sx_{i+1}
#   We approximate M ~= qscale / 2^qshift  (integer multiply + arithmetic shift),
#   which is exactly what cheap FPGA datapaths do.
# --------------------------------------------------------------------------
def gather_calib_batch(calib_imgs, sz, dev):
    tf = transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor()])
    from PIL import Image
    xs = []
    for p in calib_imgs:
        xs.append(tf(Image.open(p).convert("RGB")))
    return torch.stack(xs).to(dev)

def percentile_absmax(t, pct=99.9):
    a = t.abs().flatten()
    k = max(1, int(a.numel() * pct / 100.0))
    return torch.kthvalue(a, k).values.item()

def quantize_export(model, calib_imgs, cfg, dev):
    sz = cfg["img_size"]
    layers = collect_folded_layers(model)
    fc_w = model.fc.weight.detach().clone()
    fc_b = model.fc.bias.detach().clone()
    folded = FoldedNet(layers, fc_w, fc_b).to(dev).eval()

    # ---- 1. calibration. Record conv INPUT and post-ReLU OUTPUT activations.
    #         Activations are UNSIGNED (post_proc clamps ReLU to [0,255]), so
    #         act scale = max_value / 255. Weights stay signed -> /127. ----
    xb = gather_calib_batch(calib_imgs, sz, dev)
    with torch.no_grad():
        folded(xb, record=True)
    acts = folded.acts  # [img, relu0, (pool0?), relu1, (pool1?), ...]

    # Walk the recorded list using each layer's pool flag (variable positions).
    conv_in_acts, conv_out_acts = [], []
    ai = 0
    for l in layers:
        conv_in_acts.append(acts[ai])       # input to this conv
        conv_out_acts.append(acts[ai + 1])  # post-ReLU output
        ai += 1                             # advance past conv output
        if l["pool"]:
            ai += 1                         # advance past pool output (scale unchanged)

    # Unsigned activation scales: value in [0, max] maps to int [0,255].
    def act_scale(t):  # 99.9th percentile of the (non-negative) activations
        a = t.flatten()
        k = max(1, int(a.numel() * 99.9 / 100.0))
        return max(torch.kthvalue(a, k).values.item() / 255.0, 1e-8)
    # input image is in [0,1]; quantize to [0,255] => scale = 1/255 nominal,
    # but use measured max for headroom.
    in_scales  = [act_scale(a) for a in conv_in_acts]
    out_scales = [act_scale(a) for a in conv_out_acts]
    for i in range(1, len(in_scales)):       # chain consistency
        in_scales[i] = out_scales[i - 1]

    # ---- 2. quantize weights/bias; requant maps int32 acc -> uint8 at s_out.
    #         post_proc computes (acc*qscale)>>qshift, qscale is 16-bit UNSIGNED
    #         (<=65535), qshift is 5-bit (0..31). We pick the largest qshift in
    #         range that keeps qscale<=65535, to preserve precision. ----
    export = []
    for i, l in enumerate(layers):
        w = l["weight"]; b = l["bias"]
        w_absmax = w.abs().max().item()
        sw = (w_absmax / 127.0) if w_absmax > 0 else 1e-8
        w_q = torch.clamp(torch.round(w / sw), -127, 127).to(torch.int8)
        sx = in_scales[i]
        s_acc = sx * sw                          # int32 accumulator scale
        b_q = torch.round(b / s_acc).to(torch.int32)
        s_out = out_scales[i]
        M = s_acc / s_out                        # requant multiplier
        # choose qshift (<=31) so qscale = round(M<<qshift) fits in 16 bits
        qshift = 0
        for sh in range(31, -1, -1):
            if round(M * (1 << sh)) <= 65535:
                qshift = sh; break
        qscale = max(1, min(65535, int(round(M * (1 << qshift)))))
        export.append(dict(
            **l, w_q=w_q, b_q=b_q, sx=sx, sw=sw, s_acc=s_acc,
            s_out=s_out, qscale=qscale, qshift=qshift,
        ))

    last_s_acc = export[-1]["s_out"]

    # ---- 3. write files ----
    os.makedirs(cfg["out_dir"], exist_ok=True)
    wpath = os.path.join(cfg["out_dir"], "weights.bin")
    bpath = os.path.join(cfg["out_dir"], "biases.bin")
    hpath = os.path.join(cfg["out_dir"], "model_arch.h")

    woff = boff = 0
    rows = []
    with open(wpath, "wb") as fw, open(bpath, "wb") as fb:
        for i, e in enumerate(export):
            wb_bytes = e["w_q"].cpu().numpy().tobytes()       # int8, OIHW order
            bb_bytes = e["b_q"].cpu().numpy().tobytes()       # int32
            fw.write(wb_bytes); fb.write(bb_bytes)
            wsz, bsz = len(wb_bytes), len(bb_bytes)
            rows.append((e, woff, wsz, boff, bsz))
            woff += wsz; boff += bsz

    fc_w_np = fc_w.cpu().numpy().astype(np.float32)  # [2, Clast]
    fc_b_np = fc_b.cpu().numpy().astype(np.float32)  # [2]

    with open(hpath, "w") as h:
        h.write("#ifndef MODEL_ARCH_H\n#define MODEL_ARCH_H\n#include <stdint.h>\n\n")
        h.write("/* Auto-generated. Matches the broadcast INT8 accelerator RTL:\n"
                "   - activations are UNSIGNED uint8 [0,255] (post_proc ReLU clamp)\n"
                "   - weights signed int8 [-127,127]\n"
                "   - requant: out = clamp((acc*qscale) >> qshift, 0, 255)\n"
                "   - qscale is 16-bit unsigned (<=65535), qshift 0..31\n"
                "   - per layer: C_out<=128 (NUM_PE), Cin*K*K<=512 (weight BRAM)\n"
                "   - conv on the array; bias add / requant / ReLU / pool / GAP / FC on CPU */\n\n")
        h.write(f"#define INPUT_SIZE     {sz}\n")
        h.write(f"#define INPUT_CH       3\n")
        h.write(f"#define NUM_CLASSES    2\n")
        h.write(f"#define LAST_CONV_CH   {export[-1]['out_ch']}\n")
        h.write(f"static const float INPUT_SCALE = {export[0]['sx']:.10e}f;"
                f"  /* q = round((px/255)/INPUT_SCALE), clamp [0,255] */\n")
        h.write(f'static const char* CLASS_NAMES[NUM_CLASSES] = {{"defective","non-defective"}};\n\n')
        h.write("typedef struct {\n")
        h.write("    uint32_t in_ch, out_ch;\n")
        h.write("    uint32_t k_h, k_w;\n")
        h.write("    uint32_t stride, pad;\n")
        h.write("    uint32_t pool;                         /* 1 = 2x2 maxpool after this conv */\n")
        h.write("    uint32_t relu;                         /* 1 = ReLU (cfg_relu) */\n")
        h.write("    uint32_t weight_offset, weight_size;   /* bytes into weights.bin (int8) */\n")
        h.write("    uint32_t bias_offset,   bias_size;     /* bytes into biases.bin  (int32) */\n")
        h.write("    uint32_t qscale;  uint32_t qshift;     /* out = (acc*qscale)>>qshift */\n")
        h.write("} layer_config_t;\n\n")
        h.write(f"#define NUM_LAYERS {len(rows)}\n\n")
        h.write("static const layer_config_t model_layers[NUM_LAYERS] = {\n")
        for (e, wo, ws, bo, bs) in rows:
            h.write(f"    {{{e['in_ch']}, {e['out_ch']}, {e['k']}, {e['k']}, "
                    f"{e['stride']}, {e['pad']}, {1 if e['pool'] else 0}, 1, "
                    f"{wo}, {ws}, {bo}, {bs}, "
                    f"{e['qscale']}, {e['qshift']}}},\n")
        h.write("};\n\n")
        h.write(f"static const float LAST_ACC_SCALE = {last_s_acc:.10e}f;\n\n")
        h.write(f"static const float FC_WEIGHT[NUM_CLASSES][LAST_CONV_CH] = {{\n")
        for r in range(fc_w_np.shape[0]):
            vals = ", ".join(f"{v:.8e}f" for v in fc_w_np[r])
            h.write(f"    {{{vals}}},\n")
        h.write("};\n")
        h.write(f"static const float FC_BIAS[NUM_CLASSES] = {{"
                + ", ".join(f"{v:.8e}f" for v in fc_b_np) + "};\n\n")
        h.write("#endif /* MODEL_ARCH_H */\n")

    # save quantized state for the record
    torch.save({"export": [{k: (v.cpu() if torch.is_tensor(v) else v)
                            for k, v in e.items()} for e in export],
                "fc_w": fc_w.cpu(), "fc_b": fc_b.cpu()},
               os.path.join(cfg["out_dir"], "model_int8.pt"))

    return export, folded, last_s_acc, fc_w, fc_b

# --------------------------------------------------------------------------
# INT8 reference simulation (matches the C code) to verify accuracy
# --------------------------------------------------------------------------
def int8_infer(export, last_s_acc, fc_w, fc_b, x_float):
    """Replicate the exact integer datapath the FPGA + C run.
    Activations are UNSIGNED [0,255] (post_proc). Requant uses an arithmetic
    right shift: out = clamp((acc * qscale) >> qshift, 0, 255).
    Runs entirely on CPU (tiny integer math) so it is device-agnostic."""
    x_float = x_float.detach().to("cpu")
    fc_w = fc_w.detach().to("cpu"); fc_b = fc_b.detach().to("cpu")
    # quantize input image to [0,255] at layer 0's input scale
    xq = torch.clamp(torch.round(x_float / export[0]["sx"]), 0, 255)
    for e in export:
        w_q = e["w_q"].detach().to("cpu").float()
        b_q = e["b_q"].detach().to("cpu").float()
        acc = F.conv2d(xq, w_q, None, stride=e["stride"], padding=e["pad"])
        acc = acc + b_q.reshape(1, -1, 1, 1)                        # int32 accumulator
        # post_proc: (acc * qscale) >>> qshift  (floor = arithmetic shift for >=0)
        scaled = acc * e["qscale"]
        out = torch.floor(scaled / (1 << e["qshift"]))
        out = torch.clamp(out, 0, 255)                              # ReLU + uint8 clamp
        xq = out
        if e["pool"]:
            xq = F.max_pool2d(xq, 2)                                # uint8 maxpool
    feat = F.adaptive_avg_pool2d(xq, 1).flatten(1) * last_s_acc      # dequant -> float
    logits = F.linear(feat, fc_w, fc_b)                            # float FC on CPU
    return logits

# --------------------------------------------------------------------------
# Export sample test images as raw uint8 HWC .bin files for the FPGA.
#   Format: INPUT_SIZE*INPUT_SIZE*3 bytes, row-major, channel-last (R,G,B),
#   uint8 [0,255].  Both Python and C do: real = (uint8/255); q = real/INPUT_SCALE
# --------------------------------------------------------------------------
def export_test_bins(test_ds, export, cfg, n=4):
    from PIL import Image
    sz = cfg["img_size"]
    bin_dir = os.path.join(cfg["out_dir"], "test_bins")
    os.makedirs(bin_dir, exist_ok=True)
    rng = random.Random(cfg["seed"] + 1)
    samples = list(test_ds.samples); rng.shuffle(samples)
    picked = samples[:n]
    for k, (path, lab) in enumerate(picked):
        img = Image.open(path).convert("RGB").resize((sz, sz))
        arr = np.asarray(img, dtype=np.uint8)              # HWC uint8
        arr.tofile(os.path.join(bin_dir, f"sample_{k}_{CLASSES[lab]}.bin"))
    print(f"[INFO] Wrote {len(picked)} test .bin images -> {bin_dir}/ "
          f"(uint8 HWC {sz}x{sz}x3)")
    return bin_dir, picked

# --------------------------------------------------------------------------
# CALIB folder
# --------------------------------------------------------------------------
def make_calib(train_ds, cfg):
    os.makedirs(cfg["calib_dir"], exist_ok=True)
    # balanced sample across both classes
    by_cls = {0: [], 1: []}
    for path, lab in train_ds.samples:
        by_cls[lab].append(path)
    per = cfg["n_calib"] // 2
    chosen = []
    rng = random.Random(cfg["seed"])
    for c in (0, 1):
        rng.shuffle(by_cls[c]); chosen += by_cls[c][:per]
    for i, src in enumerate(chosen):
        ext = os.path.splitext(src)[1]
        shutil.copy(src, os.path.join(cfg["calib_dir"], f"calib_{i:03d}{ext}"))
    print(f"[INFO] Wrote {len(chosen)} calibration images -> {cfg['calib_dir']}/")
    return chosen

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    cfg = vars(ap.parse_args())
    set_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={dev}  img_size={cfg['img_size']}")

    tr_dl, va_dl, te_dl, tr_ds, va_ds, te_ds = build_loaders(cfg)
    print(f"[INFO] train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} classes={tr_ds.classes}")

    model = RailNet().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] RailNet params={n_params:,}")

    model = train(model, tr_dl, va_dl, cfg, dev)
    acc, prec, rec = evaluate(model, te_dl, dev)
    print(f"\n[FP32 TEST] acc={acc:.3f} precision(defective)={prec:.3f} recall(defective)={rec:.3f}")

    calib_imgs = make_calib(tr_ds, cfg)
    export, folded, last_s_acc, fc_w, fc_b = quantize_export(model, calib_imgs, cfg, dev)

    # ---- hardware-constraint guarantee report (against the RTL limits) ----
    print("\n[HW-CHECK] verifying every layer against the accelerator RTL:")
    sz = cfg["img_size"]; h = w = sz; c = 3
    ok_all = True
    if c*h*w > 32768:
        print(f"  input {c}x{h}x{w}={c*h*w} > 32768  FAIL"); ok_all=False
    for i, e in enumerate(export):
        pad=e["pad"]; k=e["k"]; s=e["stride"]
        oh=(h+2*pad-k)//s+1; ow=(w+2*pad-k)//s+1
        mac=e["in_ch"]*k*k
        c1 = e["out_ch"]<=128
        c2 = mac<=512
        c3 = e["out_ch"]*oh*ow<=32768
        c4 = e["qscale"]<=65535 and e["qshift"]<=31
        c5 = max(oh,ow)<=127
        ok = c1 and c2 and c3 and c4 and c5; ok_all &= ok
        print(f"  L{i}: Cin={e['in_ch']:3d} Cout={e['out_ch']:3d} {h}x{w}->{oh}x{ow} "
              f"mac={mac:3d} qscale={e['qscale']:5d} qshift={e['qshift']:2d} "
              f"out={e['out_ch']*oh*ow:5d}  {'OK' if ok else 'FAIL'}")
        h,w,c = oh,ow,e["out_ch"]
        if e["pool"]: h//=2; w//=2
    print(f"  ==> {'ALL CONSTRAINTS SATISFIED' if ok_all else 'CONSTRAINT VIOLATION (see above)'}")

    # verify INT8 vs FP32 on the test set
    correct = tot = 0
    for x, y in te_dl:
        for j in range(x.size(0)):
            logit = int8_infer(export, last_s_acc, fc_w, fc_b, x[j:j+1])
            correct += int(logit.argmax(1).item() == y[j].item()); tot += 1
    print(f"[INT8 TEST] acc={correct/max(tot,1):.3f}  (simulated integer datapath)")

    # export a few real test images as .bin and print expected predictions,
    # reading them back exactly as the C code will (uint8 HWC / 255)
    bin_dir, picked = export_test_bins(te_ds, export, cfg)
    sz = cfg["img_size"]
    print("\n[VERIFY] Expected predictions per exported .bin (use these to check the FPGA):")
    for k, (path, lab) in enumerate(picked):
        raw = np.fromfile(os.path.join(bin_dir, f"sample_{k}_{CLASSES[lab]}.bin"),
                          dtype=np.uint8).astype(np.float32) / 255.0
        t = torch.from_numpy(raw.reshape(sz, sz, 3)).permute(2, 0, 1).unsqueeze(0)
        logit = int8_infer(export, last_s_acc, fc_w, fc_b, t)
        pred = int(logit.argmax(1).item())
        print(f"  sample_{k}: true={CLASSES[lab]:13s} pred={CLASSES[pred]:13s} "
              f"logits={logit.detach().numpy().round(3).tolist()}")
    print(f"\n[DONE] Exported to '{cfg['out_dir']}/':  weights.bin  biases.bin  "
          f"model_arch.h  model_int8.pt  test_bins/")

if __name__ == "__main__":
    main()
