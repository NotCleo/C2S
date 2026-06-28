/* =====================================================================
 *  Aerial Vehicle Detector — INT8 inference on archway_npu (Arty A7-100T)
 *    default build : pure-software conv (validation, runs anywhere)
 *    -DUSE_FPGA    : drives the archway_npu over /dev/mem (AXI-Lite)
 *
 *  Pipeline (matches new_train.py byte-for-byte):
 *    full image (CANVAS x CANVAS x3 uint8 HWC, from new_img2bin.py)
 *      -> split into TILE x TILE tiles
 *      -> per tile: quantize to [0,127]; run the conv backbone (NPU);
 *                   read the CfxGxG feature map; float 1x1 HEAD on CPU;
 *                   decode anchor-free boxes; offset to canvas coords
 *      -> global per-class NMS
 *      -> coordinates.csv  (x_min,y_min,x_max,y_max,score,class_id,class_name)
 *
 *  Register map (decoded from archway_npu_v1_0_S00_AXI.v / cnn_top.v — the
 *  SAME map the proven railway classifier used):
 *    NPU base = 0x10140000 (256K window)
 *      0x00 cfg_cin   0x04 cfg_cout 0x08 cfg_hin  0x0C cfg_win
 *      0x10 cfg_hout  0x14 cfg_wout 0x18 cfg_k    0x1C cfg_stride
 *      0x20 cfg_pad   0x24 cfg_qscale 0x28 cfg_qshift 0x2C cfg_relu
 *      0x30 layer_start  0x34 READ layer_done  0x38 READ pp_busy
 *    Weight BRAM at base+0x10000 (awaddr[17:16]=01), 32-bit words.
 *    Act buffer  at base+0x20000 (awaddr[17:16]=10), byte addressed, bit14=bank.
 *
 *  HARDWARE TRUTHS (src/post_proc.v, src/pe.v) — must be obeyed:
 *    * NO BIAS: post_proc does out = clip((acc*qscale) >> qshift, 0,255).
 *      biases.bin is read for the file contract but NEVER added on the FPGA
 *      path (the software fallback also drops it, so SW == HW == board).
 *    * pe.v multiplies $signed(pixel_in): activations are read back SIGNED.
 *      We keep them in [0,127]; the software fallback reinterprets any byte
 *      >=128 as (byte-256) exactly like the silicon.
 *
 *  Build (sw):   gcc -O2 -Wall -o new_infer new_infer.c -lm
 *  Build (fpga): riscv32-linux-gcc new_infer.c -o new_infer -static -lm -DUSE_FPGA
 * ===================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "model_arch.h"

#ifdef USE_FPGA
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

/* The archway_npu writes its conv output channel-LAST (HWC) but reads its input
 * channel-MAJOR (CHW).  Verified against the RTL (the authoritative source):
 *   layer_ctrl.v input  read : e_act_addr = cin*(Hin*Win) + y*Win + x   -> CHW
 *   cnn_top.v    output write: act_wr_addr = (h*Wout+w)*Cout + chan      -> HWC
 * So the host must TRANSPOSE on read-back (HWC->CHW); the write side stays CHW.
 * The railway classifier never noticed this because its global-average-pool +
 * argmax head is invariant to the transpose; a per-cell detector is NOT -- the
 * untransposed read scrambles every (channel,gy,gx) feature and floods boxes.
 * Hence this MUST be 1 for the detector. (Set 0 only for a CHW board revision.) */
#define NPU_OUT_HWC 1

/* HW-vs-SW self-test: build with -DNPU_SELFTEST=1 for a one-off diagnostic run
 * that compares the FPGA conv against the bit-exact software model per layer.
 * Default 0 (off) so production runs pay nothing. */
#ifndef NPU_SELFTEST
#define NPU_SELFTEST 0
#endif

typedef struct { int32_t *data; int C, H, W; } tensor_t;
static tensor_t t_alloc(int C, int H, int W) {
    tensor_t t = { (int32_t*)calloc((size_t)C*H*W, sizeof(int32_t)), C, H, W };
    if (!t.data) { fprintf(stderr, "[FATAL] OOM %dx%dx%d\n", C, H, W); exit(1); }
    return t;
}
static void t_free(tensor_t *t) { free(t->data); t->data = NULL; }
#define IDX(t,c,y,x) ((((c)*(t).H)+(y))*(t).W+(x))

/* ---------- software conv (ground truth + fallback) ----------------------
 * Mirrors the NPU exactly: activations read as SIGNED int8, NO bias, and
 * requant out = clip((acc*qscale) >> qshift, 0, 255). */
__attribute__((unused))
static void conv_hw_sw(const tensor_t in, const int8_t *w, int out_ch, int k,
                       int stride, int pad, uint32_t qscale, uint32_t qshift,
                       tensor_t *out) {
    int OH = (in.H + 2*pad - k)/stride + 1, OW = (in.W + 2*pad - k)/stride + 1;
    *out = t_alloc(out_ch, OH, OW);
    for (int oc = 0; oc < out_ch; ++oc)
      for (int oy = 0; oy < OH; ++oy)
        for (int ox = 0; ox < OW; ++ox) {
          int32_t acc = 0;
          for (int ic = 0; ic < in.C; ++ic) {
            const int8_t *wk = w + (((oc*in.C)+ic)*k*k);
            for (int ky = 0; ky < k; ++ky) { int iy = oy*stride+ky-pad; if (iy<0||iy>=in.H) continue;
              for (int kx = 0; kx < k; ++kx) { int ix = ox*stride+kx-pad; if (ix<0||ix>=in.W) continue;
                int32_t v = in.data[IDX(in,ic,iy,ix)] & 0xFF;       /* stored byte */
                int32_t s = (v >= 128) ? (v - 256) : v;             /* SIGNED int8 (pe.v) */
                acc += s * (int32_t)wk[ky*k+kx];
              }}}
          int64_t r = ((int64_t)acc * (int64_t)qscale) >> qshift;   /* arithmetic shift */
          if (r < 0) r = 0;                                         /* ReLU */
          if (r > 255) r = 255;                                     /* uint8 clamp */
          out->data[IDX((*out),oc,oy,ox)] = (int32_t)r;
        }
}

static tensor_t maxpool2(const tensor_t in) {           /* unsigned max, like the host CPU */
    tensor_t o = t_alloc(in.C, in.H/2, in.W/2);
    for (int c = 0; c < in.C; ++c) for (int y = 0; y < o.H; ++y) for (int x = 0; x < o.W; ++x) {
        int32_t m = in.data[IDX(in,c,2*y,2*x)];
        int32_t a = in.data[IDX(in,c,2*y,2*x+1)];
        int32_t b = in.data[IDX(in,c,2*y+1,2*x)];
        int32_t d = in.data[IDX(in,c,2*y+1,2*x+1)];
        if (a>m) m=a;
        if (b>m) m=b;
        if (d>m) m=d;
        o.data[IDX(o,c,y,x)] = m;
    }
    return o;
}

/* =====================================================================
 *  HARDWARE PATH (AXI-Lite over /dev/mem) — copied from the proven infer.c
 * ===================================================================== */
#ifdef USE_FPGA
#define NPU_BASE   0x10140000u
#define NPU_SPAN   (256u*1024u)
#define OFF_WEIGHT 0x10000u
#define OFF_ACT    0x20000u
#define ACT_BANKBIT (1u<<14)

#define R_CIN 0x00
#define R_COUT 0x04
#define R_HIN 0x08
#define R_WIN 0x0C
#define R_HOUT 0x10
#define R_WOUT 0x14
#define R_K 0x18
#define R_STRIDE 0x1C
#define R_PAD 0x20
#define R_QSCALE 0x24
#define R_QSHIFT 0x28
#define R_RELU 0x2C
#define R_START 0x30
#define R_DONE 0x34
#define R_PPBUSY 0x38

static volatile uint8_t *g_base;
static inline void w32(uint32_t off, uint32_t v) { *(volatile uint32_t*)(g_base+off) = v; }
static inline uint32_t r32(uint32_t off) { return *(volatile uint32_t*)(g_base+off); }

static int npu_map(void) {
    int fd = open("/dev/mem", O_RDWR|O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return -1; }
    void *p = mmap(NULL, NPU_SPAN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, NPU_BASE);
    if (p == MAP_FAILED) { perror("mmap"); close(fd); return -1; }
    g_base = (volatile uint8_t*)p; return 0;
}

static void npu_load_weights(const int8_t *w, int in_ch, int out_ch, int k) {
    int need_sub = (out_ch + 3) >> 2;
    for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) for (int cin = 0; cin < in_ch; ++cin) {
        int r = (kh*k+kw)*in_ch + cin;
        volatile uint32_t *row = (volatile uint32_t*)(g_base+OFF_WEIGHT+((uint32_t)(r*32)<<2));
        for (int sub = 0; sub < need_sub; ++sub) {
            uint32_t word = 0;
            for (int b = 0; b < 4; ++b) {
                int oc = sub*4+b; int8_t wv = 0;
                if (oc < out_ch) wv = w[((oc*in_ch+cin)*k+kh)*k+kw];   /* OIHW */
                word |= ((uint32_t)(uint8_t)wv) << (8*b);
            }
            row[sub] = word;
        }
    }
}

/* Activation I/O: ONE byte per AXI-Lite access, matching the proven railway
 * driver.  A word-packed (4 bytes / 32-bit transaction) version was tried for
 * speed, but the archway_npu activation port does NOT round-trip full-word
 * writes -- the self-test showed it garbles layer 1 already (hw=228 vs sw=2),
 * cascading into the box flood.  Byte-wise is the known-good HW contract; we
 * take correctness over the ~4x transaction count. */
static void npu_write_acts(const tensor_t in, int bank) {       /* CHW bytes */
    volatile uint8_t *ab = (volatile uint8_t*)(g_base+OFF_ACT+(bank?ACT_BANKBIT:0));
    int n = in.C*in.H*in.W;
    for (int i = 0; i < n; ++i) ab[i] = (uint8_t)in.data[i];     /* direct byte store */
}
static void npu_read_acts(tensor_t *out, int bank) {
    volatile uint8_t *ab = (volatile uint8_t*)(g_base+OFF_ACT+(bank?ACT_BANKBIT:0));
#if NPU_OUT_HWC
    /* hardware wrote (h*W+w)*C+c -> transpose to CHW */
    for (int y = 0; y < out->H; ++y) for (int x = 0; x < out->W; ++x) for (int c = 0; c < out->C; ++c)
        out->data[IDX((*out),c,y,x)] = (int32_t)(uint8_t)ab[(y*out->W+x)*out->C + c];
#else
    int n = out->C*out->H*out->W;
    for (int i = 0; i < n; ++i) out->data[i] = (int32_t)(uint8_t)ab[i];  /* single byte read */
#endif
}

static int g_layer = 0;

/* Load a layer's weights into the BRAM ONCE; they stay resident while we push
 * every tile through this same layer (layer-major loop), instead of re-pushing
 * 124KB over the bus for all 36 tiles. */
static void npu_set_weights(const int8_t *w, int in_ch, int out_ch, int k) {
    npu_load_weights(w, in_ch, out_ch, k);
}

/* Run ONE conv with weights already resident. Keeps the per-conv bank toggle
 * (g_layer&1): the HW flips its internal read/write bank on every layer_start
 * pulse, so the host parity must advance once per conv -- which it still does,
 * regardless of tile/layer ordering, because we write the input and read the
 * output in full every call (no cross-conv reliance on bank contents). */
static void npu_run_conv(const tensor_t in, int out_ch, int k,
                         int stride, int pad, uint32_t qscale, uint32_t qshift,
                         uint32_t relu, tensor_t *out) {
    int OH = (in.H + 2*pad - k)/stride + 1, OW = (in.W + 2*pad - k)/stride + 1;
    *out = t_alloc(out_ch, OH, OW);
    int in_bank = g_layer & 1, out_bank = in_bank ^ 1;
    npu_write_acts(in, in_bank);
    w32(R_CIN, in.C); w32(R_COUT, out_ch);
    w32(R_HIN, in.H); w32(R_WIN, in.W);
    w32(R_HOUT, OH);  w32(R_WOUT, OW);
    w32(R_K, k); w32(R_STRIDE, stride); w32(R_PAD, pad);
    w32(R_QSCALE, qscale); w32(R_QSHIFT, qshift); w32(R_RELU, relu);

    /* Proven poll logic (do NOT shorten the guard or re-pulse START): while the
     * MAC array is computing, pp_busy is still 0 and these polls are fast, so a
     * heavy 504-MAC layer legitimately needs MANY iterations before pp_busy
     * rises.  A short guard mistakes that for a missed start; re-pulsing START
     * mid-compute corrupts the conv (it broke layers 3-6 -> 0 detections).  The
     * 20M guard comfortably covers the slowest layer; a genuinely missed start
     * is rare (intermittent) and just warns. */
    /* PROVEN sequence -- do NOT add a usleep/settle, do NOT re-pulse START, do NOT
     * skip the read.  Each of those (tried) breaks exactly the k3 s1 heavy convs
     * (layers 3-6): anything inserted between the previous conv and this start
     * pulse makes the NPU drop the start for those configs.  This untouched
     * sequence runs all 8 layers correctly (a clean run matches the CPU sim). */
    int bad = 0;
    w32(R_START, 0);
    { long g2 = 0; while ((r32(R_PPBUSY)&1u) != 0) { if (++g2 > 20000000) break; } }
    w32(R_START, 0); w32(R_START, 1);                          /* rising edge triggers */

    /* WHOLE-LAYER completion (RTL-verified).  pp_busy pulses once PER OUTPUT
     * POSITION: layer_ctrl.v asserts pp_start every spatial position and post_proc
     * raises pp_busy for each 128-ch drain -- so it toggles Hout*Wout times (1024
     * for L1), NOT once per layer.  The old "first rise then first fall" detected
     * only position 0 and let the host read / re-arm while the layer was still
     * computing -> nondeterministic output (selftest HWdet up to 234, worst on the
     * 1024-position L1).  R_DONE (0x34) is a 1-cycle pulse in the AXI shim, so a
     * slow core can't poll it.  Detect completion the TIMING-INDEPENDENT way:
     *   (1) confirm the layer actually started -- catch any pp_busy high (or the
     *       rare R_DONE) so we never mistake the output bank's pre-existing data
     *       for a result;
     *   (2) read the output until two consecutive FULL reads are identical.  A
     *       full read takes far longer than the per-position write interval, so
     *       while positions are still draining the reads differ; once the layer is
     *       truly done the buffer is frozen and they match.  The act-buffer read
     *       path is a registered BRAM (deterministic), so a frozen buffer reads
     *       identically -- "stable" means "layer finished", in any clock regime. */
    { long guard = 0; int started = 0;
      while (!started) {
        if (r32(R_DONE)&1u)   { started = 1; break; }          /* end pulse: it ran */
        if (r32(R_PPBUSY)&1u) { started = 1; break; }          /* a drain pulse: running */
        if (++guard > 20000000) { fprintf(stderr, "[HW] conv %d never started -> tile zeroed\n", g_layer); bad = 1; break; }
      }
    }
    w32(R_START, 0);                                           /* clear for next rising edge */
    if (!bad) {
        int n = out->C * out->H * out->W;
        int32_t *prev = (int32_t*)malloc((size_t)n * sizeof(int32_t));
        npu_read_acts(out, out_bank);
        if (prev) {
            int stable = 0, tries = 0;
            while (!stable && tries++ < 16) {
                memcpy(prev, out->data, (size_t)n * sizeof(int32_t));
                npu_read_acts(out, out_bank);
                stable = (memcmp(prev, out->data, (size_t)n * sizeof(int32_t)) == 0);
            }
            free(prev);
            if (!stable) fprintf(stderr, "[HW] conv %d output never settled (still draining?)\n", g_layer);
        }
    }
    /* On a genuine miss (never started) discard the read -> benign zeros instead of
     * a garbage flood (out->data is calloc'd, so it is already zero). */
    if (bad) memset(out->data, 0, (size_t)out->C * out->H * out->W * sizeof(int32_t));
    g_layer++;
}
#endif /* USE_FPGA */

/* ---------- run one conv layer (HW or SW) -> uint8 [0,255] tensor ---------
 * HW: weights must already be loaded via npu_set_weights (w unused here).
 * SW: full software conv reads w directly. */
static tensor_t run_conv(const tensor_t x, layer_config_t L, const int8_t *w) {
    tensor_t acc;
#ifdef USE_FPGA
    (void)w;
    npu_run_conv(x, (int)L.out_ch, (int)L.k_h, (int)L.stride, (int)L.pad,
                 L.qscale, L.qshift, L.relu, &acc);
#else
    conv_hw_sw(x, w, (int)L.out_ch, (int)L.k_h, (int)L.stride, (int)L.pad,
               L.qscale, L.qshift, &acc);
#endif
    return acc;
}

/* ===================================================================== *
 *  DETECTION HEAD + DECODE + NMS  (float, on CPU; matches new_train.py)
 * ===================================================================== */
typedef struct { float x1, y1, x2, y2, score; int cls; } box_t;

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static float iou(box_t a, box_t b) {
    float ix1 = a.x1 > b.x1 ? a.x1 : b.x1;
    float iy1 = a.y1 > b.y1 ? a.y1 : b.y1;
    float ix2 = a.x2 < b.x2 ? a.x2 : b.x2;
    float iy2 = a.y2 < b.y2 ? a.y2 : b.y2;
    float iw = ix2 - ix1, ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.0f;
    float inter = iw * ih;
    float ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return ua > 0 ? inter/ua : 0.0f;
}

#define MAXDET 32768
static box_t g_box[MAXDET];
static int   g_nbox = 0;

static void push_box(box_t b) { if (g_nbox < MAXDET) g_box[g_nbox++] = b; }

/* per-class greedy NMS, in place; returns kept count, kept boxes at front */
static int run_nms(box_t *boxes, int n, float iou_thr) {
    /* sort by score descending (simple insertion-friendly qsort) */
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            if (boxes[j].score > boxes[i].score) { box_t t = boxes[i]; boxes[i] = boxes[j]; boxes[j] = t; }
    int *dead = (int*)calloc(n, sizeof(int));
    int keep = 0;
    for (int i = 0; i < n; ++i) {
        if (dead[i]) continue;
        boxes[keep++] = boxes[i];
        for (int j = i+1; j < n; ++j)
            if (!dead[j] && boxes[j].cls == boxes[i].cls && iou(boxes[i], boxes[j]) >= iou_thr)
                dead[j] = 1;
    }
    free(dead);
    return keep;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[DEBUG] Aerial detector starting.\n");
    if (argc < 2) { fprintf(stderr, "usage: %s image.bin [weights.bin biases.bin]\n", argv[0]); return 1; }
    const char *img_path = argv[1];
    const char *w_path = (argc > 2) ? argv[2] : "weights.bin";
    const char *b_path = (argc > 3) ? argv[3] : "biases.bin";

#ifdef USE_FPGA
    printf("[DEBUG] Mapping accelerator at 0x%08X ...\n", NPU_BASE);
    if (npu_map() != 0) { fprintf(stderr, "[FATAL] map failed (run as root?)\n"); return 1; }
    printf("[DEBUG] NPU mapped. CONV WILL RUN ON ACCELERATOR.\n");
#else
    printf("[DEBUG] SOFTWARE build (no -DUSE_FPGA). Conv runs on CPU.\n");
#endif

    /* ---- weights ---- */
    FILE *fw = fopen(w_path, "rb"); if (!fw) { perror(w_path); return 1; }
    fseek(fw, 0, SEEK_END); long wsz = ftell(fw); fseek(fw, 0, SEEK_SET);
    int8_t *W = (int8_t*)malloc(wsz);
    if (fread(W, 1, wsz, fw) != (size_t)wsz) { fprintf(stderr, "[FATAL] short read %s\n", w_path); return 1; }
    fclose(fw); printf("[DEBUG] weights.bin = %ld bytes\n", wsz);

    /* ---- biases (read for the file contract; NOT used on the HW path) ---- */
    FILE *fb = fopen(b_path, "rb");
    if (fb) { fseek(fb, 0, SEEK_END); long bsz = ftell(fb); fclose(fb);
              printf("[DEBUG] biases.bin = %ld bytes (HW ignores bias)\n", bsz); }
    else    { printf("[DEBUG] biases.bin not found (ok, HW ignores bias)\n"); }

    /* ---- image: CANVAS x CANVAS x 3 uint8 HWC ---- */
    long npx = (long)CANVAS * CANVAS * INPUT_CH;
    FILE *fi = fopen(img_path, "rb"); if (!fi) { perror(img_path); return 1; }
    fseek(fi, 0, SEEK_END); long isz = ftell(fi); fseek(fi, 0, SEEK_SET);
    if (isz != npx) { fprintf(stderr, "[FATAL] %s is %ld bytes, expected %ld (=%dx%dx%d)\n",
                              img_path, isz, npx, CANVAS, CANVAS, INPUT_CH); return 1; }
    uint8_t *raw = (uint8_t*)malloc(npx);
    if (fread(raw, 1, npx, fi) != (size_t)npx) { fprintf(stderr, "[FATAL] bad image\n"); return 1; }
    fclose(fi); printf("[DEBUG] image = %ld bytes OK (%dx%d)\n", isz, CANVAS, CANVAS);

    printf("[CFG] CONF_THRESH=%.3f  NMS_IOU=%.3f  (compiled into THIS binary; "
           "if box count is huge, you are running a stale build)\n", CONF_THRESH, NMS_IOU);
    printf("[DEBUG] tiling %dx%d into %d x %d tiles of %d ...\n",
           CANVAS, CANVAS, N_TILES, N_TILES, TILE);

    /* ===================== TILED BACKBONE (LAYER-MAJOR) =====================
     * Build all NT tiles, then loop LAYER-outer: load each layer's weights ONCE
     * and push all tiles through it (weights load 8x total, not 8*NT). Tiles are
     * independent until NMS, so this is numerically identical to tile-major. */
    const int NT = N_TILES * N_TILES;
    tensor_t *xs   = (tensor_t*)malloc((size_t)NT * sizeof(tensor_t));
    int      *t_ox = (int*)malloc((size_t)NT * sizeof(int));
    int      *t_oy = (int*)malloc((size_t)NT * sizeof(int));
    if (!xs || !t_ox || !t_oy) { fprintf(stderr, "[FATAL] OOM tile arrays\n"); return 1; }

    { int idx = 0;
      for (int ty = 0; ty < N_TILES; ++ty) for (int tx = 0; tx < N_TILES; ++tx) {
        int ox = tx*TILE, oy = ty*TILE;
        t_ox[idx] = ox; t_oy[idx] = oy;
        tensor_t x = t_alloc(INPUT_CH, TILE, TILE);     /* CHW, quantized [0,127] */
        for (int y = 0; y < TILE; ++y) for (int xx = 0; xx < TILE; ++xx) for (int c = 0; c < INPUT_CH; ++c) {
            float real = (float)raw[((oy+y)*CANVAS + (ox+xx))*INPUT_CH + c] / 255.0f;
            int q = (int)lroundf(real / INPUT_SCALE);
            if (q < 0) q = 0;
            if (q > 127) q = 127;
            x.data[IDX(x,c,y,xx)] = q;
        }
        xs[idx++] = x;
      }
    }

#if defined(USE_FPGA) && NPU_SELFTEST
    /* ---------------------------------------------------------------------
     * HW-vs-SW self-test v2 (diagnostic; build with -DNPU_SELFTEST=1).
     *  (a) prints a BUILD stamp + the activation I/O mode, so we KNOW exactly
     *      which binary is running -- removes stale-binary ambiguity;
     *  (b) tests every layer on a CLEAN (software) input, advancing with the
     *      software result -- so a divergence is a genuine per-layer HW bug,
     *      NOT garbage inherited from an earlier layer (cascade);
     *  (c) runs each HW conv TWICE and reports max|hw1-hw2| -- exposes a
     *      handshake/read RACE (nondeterministic HW);
     *  (d) splits the per-layer error into lower|upper output halves -- a much
     *      bigger upper-half error flags the >=16384-byte bank-boundary
     *      (bit14) as the cause (our L1/L7/L8 outputs are exactly 16384 B). */
    printf("[BUILD] %s %s | acts I/O = BYTE-WISE (1 byte/AXI) | readback = %s\n",
           __DATE__, __TIME__, NPU_OUT_HWC ? "HWC->CHW transpose (RTL-correct)"
                                           : "CHW direct (WRONG for this RTL)");
    {
        int probe[3] = { 0, NT/2, NT-1 };
        printf("[SELFTEST] HW vs SW, per-layer on CLEAN input, probe tiles %d %d %d\n",
               probe[0], probe[1], probe[2]);
        for (int pi = 0; pi < 3; ++pi) {
            int ti = probe[pi];
            tensor_t cur = t_alloc(xs[ti].C, xs[ti].H, xs[ti].W);
            memcpy(cur.data, xs[ti].data,
                   (size_t)xs[ti].C*xs[ti].H*xs[ti].W*sizeof(int32_t));
            for (int li = 0; li < NUM_LAYERS; ++li) {
                layer_config_t L = model_layers[li];
                const int8_t *w = W + L.weight_offset;
                tensor_t hwo, hwo2, swo;
                npu_set_weights(w, (int)cur.C, (int)L.out_ch, (int)L.k_h);
                npu_run_conv(cur, (int)L.out_ch, (int)L.k_h, (int)L.stride,
                             (int)L.pad, L.qscale, L.qshift, L.relu, &hwo);
                npu_set_weights(w, (int)cur.C, (int)L.out_ch, (int)L.k_h);
                npu_run_conv(cur, (int)L.out_ch, (int)L.k_h, (int)L.stride,
                             (int)L.pad, L.qscale, L.qshift, L.relu, &hwo2);
                conv_hw_sw(cur, w, (int)L.out_ch, (int)L.k_h, (int)L.stride,
                           (int)L.pad, L.qscale, L.qshift, &swo);
                int n = hwo.C*hwo.H*hwo.W;
                int maxd=0, ndiff=0, nondet=0, dlo=0, dhi=0;
                int wdi=-1, wd_hv=0, wd_sv=0, wdmax=0;   /* worst DETERMINISTIC-divergent elem */
                for (int i = 0; i < n; ++i) {
                    int hv = hwo.data[i], sv = swo.data[i];
                    int d = hv>sv ? hv-sv : sv-hv;
                    if (d) ++ndiff;
                    if (d > maxd) maxd = d;
                    if (i >= n/2) { if (d>dhi) dhi=d; } else { if (d>dlo) dlo=d; }
                    int dd = hv>hwo2.data[i] ? hv-hwo2.data[i] : hwo2.data[i]-hv;
                    if (dd > nondet) nondet = dd;
                    if (dd == 0 && d > wdmax) { wdmax = d; wdi = i; wd_hv = hv; wd_sv = sv; }
                }
                printf("[ST t%2d L%d] %2dx%2dx%2d=%5dB  max|hw-sw|=%3d nd=%5d/%-5d  "
                       "lo|hi=%3d|%3d  HWdet(max|hw1-hw2|)=%3d  %s\n",
                       ti, li+1, hwo.C, hwo.H, hwo.W, n, maxd, ndiff, n,
                       dlo, dhi, nondet, maxd==0 ? "OK" : "*** DIVERGES ***");
                /* Isolate the DETERMINISTIC residual (hw1==hw2 but hw!=sw): the
                 * systematic datapath gap, with the read-race jitter factored out.
                 * Report it in the ACCUMULATOR domain (pre-requant), where the nature
                 * of the gap is legible -- a missing/added MAC term, a scale, a sign
                 * flip, or a wrong pixel/weight. For L1 (in_ch=3, [0,127] input, no
                 * signed wrap) also print all 27 MAC terms: the cleanest raw window. */
                if (pi == 0 && wdi >= 0) {
                    int HHW = swo.H * swo.W;
                    int oc = wdi / HHW, oy = (wdi % HHW) / swo.W, ox = wdi % swo.W;
                    long acc = 0;
                    for (int cin = 0; cin < (int)cur.C; ++cin)
                      for (int kh = 0; kh < (int)L.k_h; ++kh) { int iy = oy*(int)L.stride + kh - (int)L.pad;
                        for (int kw = 0; kw < (int)L.k_h; ++kw) { int ix = ox*(int)L.stride + kw - (int)L.pad;
                          if (iy<0||iy>=(int)cur.H||ix<0||ix>=(int)cur.W) continue;
                          int v = cur.data[IDX(cur,cin,iy,ix)] & 0xFF; int s = (v>=128)?v-256:v;
                          acc += s * w[(((oc*(int)cur.C)+cin)*(int)L.k_h+kh)*(int)L.k_h+kw];
                        } }
                    long rr = (acc*(long)L.qscale) >> L.qshift; if (rr<0) rr=0; if (rr>255) rr=255;
                    long hwacc = ((long)wd_hv << L.qshift) / (long)L.qscale;
                    printf("[DET L%d c%d y%d x%d] hw=%d sw=%d | sw_acc=%ld(->%ld) "
                           "hw~acc=%ld  diff_acc(sw-hw)=%ld  (qs=%u qsh=%u)\n",
                           li+1, oc, oy, ox, wd_hv, wd_sv, acc, rr, hwacc, acc-hwacc,
                           L.qscale, L.qshift);
                    if (li == 0) {
                        for (int cin = 0; cin < (int)cur.C; ++cin)
                          for (int kh = 0; kh < (int)L.k_h; ++kh) { int iy = oy*(int)L.stride + kh - (int)L.pad;
                            for (int kw = 0; kw < (int)L.k_h; ++kw) { int ix = ox*(int)L.stride + kw - (int)L.pad;
                              int oob = (iy<0||iy>=(int)cur.H||ix<0||ix>=(int)cur.W);
                              int wv = w[(((oc*(int)cur.C)+cin)*(int)L.k_h+kh)*(int)L.k_h+kw];
                              int s = 0, prod = 0;
                              if (!oob) { int v = cur.data[IDX(cur,cin,iy,ix)] & 0xFF; s=(v>=128)?v-256:v; prod=s*wv; }
                              printf("     (cin%d kh%d kw%d) pix=%4d * wt=%4d = %6d%s\n",
                                     cin, kh, kw, s, wv, prod, oob?" [PAD]":"");
                            } }
                    }
                }
                t_free(&hwo); t_free(&hwo2);
                if (L.pool) { tensor_t sp = maxpool2(swo); t_free(&swo); swo = sp; }
                t_free(&cur); cur = swo;          /* advance on CLEAN sw output */
            }
            t_free(&cur);
        }
        g_layer = 0;   /* reset bank parity so the real loop matches a fresh run */
        printf("[SELFTEST] done. Read: nd=#elements differing; lo|hi=worst diff in "
               "lower|upper output half (hi>>lo => bit14 bank-boundary); "
               "HWdet>0 => HW is nondeterministic (a read/handshake race).\n");
    }
#endif

    time_t t_start = time(NULL);
    for (int li = 0; li < NUM_LAYERS; ++li) {
        layer_config_t L = model_layers[li];
        const int8_t *w = W + L.weight_offset;
#ifdef USE_FPGA
        npu_set_weights(w, (int)xs[0].C, (int)L.out_ch, (int)L.k_h);   /* load ONCE per layer */
#endif
        for (int t = 0; t < NT; ++t) {
            tensor_t acc = run_conv(xs[t], L, w);
            t_free(&xs[t]); xs[t] = acc;
            if (L.pool) { tensor_t p = maxpool2(xs[t]); t_free(&xs[t]); xs[t] = p; }
        }
        printf("[layer %d/%d] k%d s%d -> %dx%dx%d  (%d tiles, %lds elapsed)\n",
               li+1, NUM_LAYERS, (int)L.k_h, (int)L.stride,
               xs[0].C, xs[0].H, xs[0].W, NT, (long)(time(NULL)-t_start));
    }

    /* float 1x1 detection head + decode, per tile, per grid cell */
    for (int t = 0; t < NT; ++t) {
        tensor_t x = xs[t];
        int ox = t_ox[t], oy = t_oy[t];
        float feat[LAST_CONV_CH];
        for (int gy = 0; gy < GRID; ++gy) {
          for (int gx = 0; gx < GRID; ++gx) {
            for (int c = 0; c < LAST_CONV_CH; ++c)
                feat[c] = (float)x.data[IDX(x,c,gy,gx)] * LAST_ACC_SCALE;

            float logit[NA];
            for (int o = 0; o < NA; ++o) {
                float a = DET_B[o];
                for (int c = 0; c < LAST_CONV_CH; ++c) a += DET_W[o][c] * feat[c];
                logit[o] = a;
            }
            float obj = sigmoidf(logit[0]);
            /* softmax over class logits [5 .. 5+NUM_CLASSES) */
            float mx = logit[5];
            for (int k = 1; k < NUM_CLASSES; ++k) if (logit[5+k] > mx) mx = logit[5+k];
            float den = 0.0f, best_p = 0.0f; int best_c = 0;
            for (int k = 0; k < NUM_CLASSES; ++k) { float e = expf(logit[5+k]-mx); den += e; }
            for (int k = 0; k < NUM_CLASSES; ++k) {
                float p = expf(logit[5+k]-mx)/den;
                if (p > best_p) { best_p = p; best_c = k; }
            }
            float score = obj * best_p;
            if (score <= CONF_THRESH) continue;

            float bx = sigmoidf(logit[1]), by = sigmoidf(logit[2]);
            float tw = logit[3] > 6.0f ? 6.0f : logit[3];
            float th = logit[4] > 6.0f ? 6.0f : logit[4];
            float cx = (gx + bx) * STRIDE + ox;
            float cy = (gy + by) * STRIDE + oy;
            float bw = expf(tw) * STRIDE, bh = expf(th) * STRIDE;
            box_t b;
            b.x1 = cx - bw/2; b.y1 = cy - bh/2;
            b.x2 = cx + bw/2; b.y2 = cy + bh/2;
            b.score = score; b.cls = best_c;
            push_box(b);
          }
        }
        t_free(&xs[t]);
    }
    free(xs); free(t_ox); free(t_oy);

    printf("[DEBUG] raw detections before NMS: %d\n", g_nbox);
    int keep = run_nms(g_box, g_nbox, NMS_IOU);
    printf("[DEBUG] detections after NMS: %d\n", keep);

    /* ---- write coordinates.csv (canvas pixel coords) ---- */
    FILE *fo = fopen("coordinates.csv", "w");
    if (!fo) { perror("coordinates.csv"); return 1; }
    fprintf(fo, "x_min,y_min,x_max,y_max,score,class_id,class_name\n");
    for (int i = 0; i < keep; ++i) {
        box_t b = g_box[i];
        int x1 = (int)(b.x1 < 0 ? 0 : (b.x1 > CANVAS ? CANVAS : b.x1));
        int y1 = (int)(b.y1 < 0 ? 0 : (b.y1 > CANVAS ? CANVAS : b.y1));
        int x2 = (int)(b.x2 < 0 ? 0 : (b.x2 > CANVAS ? CANVAS : b.x2));
        int y2 = (int)(b.y2 < 0 ? 0 : (b.y2 > CANVAS ? CANVAS : b.y2));
        fprintf(fo, "%d,%d,%d,%d,%.3f,%d,%s\n", x1, y1, x2, y2, b.score, b.cls, CLASS_NAMES[b.cls]);
    }
    fclose(fo);

    printf("\n=================================================\n");
    printf(" Aerial Vehicle Detector\n image : %s\n", img_path);
    printf(" wrote %d boxes -> coordinates.csv\n", keep);
    printf("=================================================\n");
    free(W); free(raw);
    printf("[DEBUG] clean exit.\n");
    return 0;
}
