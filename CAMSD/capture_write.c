/* ============================================================================
 * cam_sd.c  --  OV2640 capture → SD card write   (power-aware, v2)
 *
 *  WHY v2: previous run failed with "ACMD41 timeout" AFTER CMD0/CMD8 passed.
 *  That means wiring + address were fine; the card simply browned out during
 *  its init surge because the OV2640 was still fully powered (resetting the
 *  AXI SPI core does NOT power down the camera module).
 *
 *  Fixes in this version:
 *    1. SD card is initialised FIRST (phase 0), while the camera sensor is
 *       still in reset/standby and total board load is lowest. Once a card
 *       leaves idle it stays initialised for as long as it has power.
 *    2. After FIFO readout the sensor is put into hardware power-down via
 *       the ArduChip GPIO register (0x06, PWDN bit) -- the module actually
 *       sleeps instead of just having its SPI controller reset.
 *    3. Before writing, the card is probed with CMD58. If it lost state
 *       (power dip during capture), a full re-init is attempted with a
 *       much longer ACMD41 timeout (8 s) and 3 whole-init retries.
 *
 *  Hardware (confirmed from AddressSegments.csv):
 *    axi_quad_spi_0 @ 0x10114000  → SD card  (Pmod JC)
 *    axi_quad_spi_1 @ 0x10115000  → Camera   (ChipKit SPI header)
 *    axi_iic_0      @ 0x10116000  → Camera I2C
 *
 *  Outputs:
 *    - /image.bin on the board rootfs (scp-able immediately)
 *    - SD blocks 199 (header) + 200..499 (image, 300 blocks)
 *
 *  Flow:
 *    PHASE 0: init SD card (low load) → deassert CS, leave card ready
 *    PHASE 1: init camera, capture, read FIFO → RAM, save /image.bin,
 *             sensor PWDN + SPI1 reset
 *    PHASE 2: probe SD (CMD58); re-init only if needed; write 301 blocks
 *
 *  Compile (host):
 *    riscv32-linux-gcc cam_sd.c -o cam_sd -static -lm
 *    riscv32-linux-strip cam_sd
 *
 *  Run (board, as root):  ./cam_sd
 *  Then either:
 *    scp root@<board>:/image.bin .            (direct path)
 *  or run ./read_sd and scp image_from_sd.bin (proves the SD path)
 *  and feed it to your existing Python recovery script.
 * ========================================================================== */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

/* ============================================================
 * 1.  HARDWARE ADDRESSES  (AddressSegments.csv)
 * ============================================================ */
#define CAM_SPI_BASE    0x10115000UL   /* axi_quad_spi_1  (ChipKit header) */
#define SD_SPI_BASE     0x10114000UL   /* axi_quad_spi_0  (Pmod JC)        */
#define I2C_BASE        0x10116000UL   /* axi_iic_0                        */

#define MAP_SIZE        4096UL
#define MAP_MASK        (MAP_SIZE - 1)

/* ============================================================
 * 2.  XILINX AXI QUAD SPI REGISTERS (PG153)
 * ============================================================ */
#define XSPI_SRR    0x40
#define XSPI_CR     0x60
#define XSPI_SR     0x64
#define XSPI_DTR    0x68
#define XSPI_DRR    0x6C
#define XSPI_SSR    0x70

#define CR_SPE          0x0002
#define CR_MASTER       0x0004
#define CR_TXFIFO_RST   0x0020
#define CR_RXFIFO_RST   0x0040
#define CR_MANUAL_SS    0x0080
#define CR_INHIBIT      0x0100

#define SR_RX_EMPTY     0x0001

/* ============================================================
 * 3.  ARDUCAM / OV2640
 * ============================================================ */
#define ARDUCHIP_TEST1      0x00
#define ARDUCHIP_GPIO       0x06   /* sensor power control          */
#define   GPIO_RESET_MASK   0x01   /* sensor reset (active)         */
#define   GPIO_PWDN_MASK    0x02   /* sensor power-down (active)    */
#define ARDUCHIP_FIFO       0x04
#define ARDUCHIP_TRIG       0x41
#define CAP_DONE_MASK       0x08
#define ARDUCHIP_FIFO_SZ1   0x42
#define ARDUCHIP_FIFO_SZ2   0x43
#define ARDUCHIP_FIFO_SZ3   0x44
#define BURST_FIFO_READ     0x3C
#define OV2640_I2C_ADDR     0x30

/* ============================================================
 * 4.  SD CONSTANTS
 * ============================================================ */
#define SD_BLOCK_LEN        512
#define IMAGE_START_BLOCK   200u

#define CMD0    0
#define CMD1    1
#define CMD8    8
#define CMD16   16
#define CMD24   24
#define CMD55   55
#define CMD58   58
#define ACMD41  41

#define R1_IDLE     0x01
#define R1_READY    0x00
#define DATA_TOKEN  0xFE
#define DATA_ACCEPT 0x05

/* ============================================================
 * 5.  IMAGE GEOMETRY
 * ============================================================ */
#define IMG_W       320
#define IMG_H       240
#define IMG_BYTES   (IMG_W * IMG_H * 2)                                /* 153600 */
#define IMG_BLOCKS  ((IMG_BYTES + SD_BLOCK_LEN - 1) / SD_BLOCK_LEN)    /* 300    */

/* ============================================================
 * 6.  GLOBALS
 * ============================================================ */
static volatile uint32_t *cam_spi  = NULL;
static volatile uint32_t *sd_spi   = NULL;
static volatile uint32_t *i2c_base = NULL;
static int sd_is_sdhc = 0;

static uint8_t image_buf[IMG_BYTES];

/* ============================================================
 * 7.  MMAP HELPER
 * ============================================================ */
static volatile uint32_t *map_periph(uint32_t phys)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); exit(1); }
    void *base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, phys & ~MAP_MASK);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); exit(1); }
    close(fd);
    return (volatile uint32_t *)((uint8_t *)base + (phys & MAP_MASK));
}

/* ============================================================
 * 8.  GENERIC SPI PRIMITIVES (identical timing to the proven sd_spi.h)
 * ============================================================ */
static void spi_reset(volatile uint32_t *spi)
{
    *(volatile uint32_t *)((uint8_t *)spi + XSPI_SRR) = 0x0000000A;
    usleep(1000);
    *(volatile uint32_t *)((uint8_t *)spi + XSPI_SSR) = 0xFFFFFFFF;
    *(volatile uint32_t *)((uint8_t *)spi + XSPI_CR)  =
        CR_MASTER | CR_MANUAL_SS | CR_TXFIFO_RST | CR_RXFIFO_RST | CR_INHIBIT;
    *(volatile uint32_t *)((uint8_t *)spi + XSPI_CR)  =
        CR_MASTER | CR_MANUAL_SS | CR_SPE | CR_INHIBIT;
}

static void spi_cs(volatile uint32_t *spi, int assert)
{
    *(volatile uint32_t *)((uint8_t *)spi + XSPI_SSR) =
        assert ? 0xFFFFFFFE : 0xFFFFFFFF;
}

static uint8_t spi_xfer(volatile uint32_t *spi, uint8_t out)
{
    volatile uint32_t *SR  = (volatile uint32_t *)((uint8_t *)spi + XSPI_SR);
    volatile uint32_t *DTR = (volatile uint32_t *)((uint8_t *)spi + XSPI_DTR);
    volatile uint32_t *DRR = (volatile uint32_t *)((uint8_t *)spi + XSPI_DRR);
    volatile uint32_t *CR  = (volatile uint32_t *)((uint8_t *)spi + XSPI_CR);
    int g;

    while (!(*SR & SR_RX_EMPTY)) (void)*DRR;   /* drain stale RX */

    *DTR = out;
    uint32_t cr = *CR;
    *CR = cr & ~CR_INHIBIT;

    g = 0;
    while (*SR & SR_RX_EMPTY)
        if (++g > 1000000) break;

    *CR = cr | CR_INHIBIT;
    return (uint8_t)*DRR;
}

static inline uint8_t spi_rx(volatile uint32_t *spi)
{
    return spi_xfer(spi, 0xFF);
}

/* ============================================================
 * 9.  CAMERA SPI  (mirrors your proven camera.c)
 * ============================================================ */
static void cam_spi_init(void)
{
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SRR) = 0x0A;
    usleep(1000);
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_CR)  = 0x186;
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFF;
}

static uint8_t cam_xfer(uint8_t data)
{
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_DTR) = data;
    uint32_t cr = *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_CR);
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_CR) = cr & ~0x100;
    int t = 0;
    while ((*(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SR) & 0x01) != 0)
        if (++t > 100000) break;
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_CR) = cr | 0x100;
    return (uint8_t)*(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_DRR);
}

static void cam_write_reg(uint8_t addr, uint8_t val)
{
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFE;
    cam_xfer(addr | 0x80);
    cam_xfer(val);
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFF;
    usleep(100);
}

static uint8_t cam_read_reg(uint8_t addr)
{
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFE;
    cam_xfer(addr & 0x7F);
    uint8_t v = cam_xfer(0x00);
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFF;
    usleep(100);
    return v;
}

/* Real power-down: sensor PWDN via ArduChip GPIO, then reset the SPI core.
 * This is what actually drops the module's current draw. */
static void cam_powerdown(void)
{
    printf("  [cam] Asserting sensor PWDN (ArduChip GPIO 0x06)...\n");
    cam_write_reg(ARDUCHIP_GPIO, GPIO_PWDN_MASK);
    usleep(10000);

    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFF;
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_CR)  =
        CR_MASTER | CR_MANUAL_SS | CR_TXFIFO_RST | CR_RXFIFO_RST | CR_INHIBIT;
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SRR) = 0x0A;
    usleep(2000);
    printf("  [cam] Sensor sleeping, SPI1 reset.\n");
}

/* ============================================================
 * 10. I2C (blind mode, same as proven camera.c)
 * ============================================================ */
static void i2c_write(uint8_t reg, uint8_t val)
{
    *(volatile uint32_t *)((uint8_t *)i2c_base + 0x40)  = 0x0A;
    usleep(200);
    *(volatile uint32_t *)((uint8_t *)i2c_base + 0x100) = 0x01;
    *(volatile uint32_t *)((uint8_t *)i2c_base + 0x108) = 0x100 | (OV2640_I2C_ADDR << 1);
    *(volatile uint32_t *)((uint8_t *)i2c_base + 0x108) = reg;
    *(volatile uint32_t *)((uint8_t *)i2c_base + 0x108) = 0x200 | val;
    usleep(5000);
}

static void i2c_write_array(const uint8_t data[][2])
{
    for (int i = 0; !(data[i][0] == 0xFF && data[i][1] == 0xFF); i++)
        i2c_write(data[i][0], data[i][1]);
}

/* ============================================================
 * 11. OV2640 REGISTER TABLES (unchanged, proven with your Python script)
 * ============================================================ */
static const uint8_t OV2640_JPEG_INIT[][2] = {
    {0xff,0x00},{0x2c,0xff},{0x2e,0xdf},{0xff,0x01},{0x3c,0x32},{0x11,0x00},
    {0x09,0x02},{0x04,0x28},{0x13,0xe5},{0x14,0x48},{0x2c,0x0c},{0x33,0x78},
    {0x3a,0x33},{0x3b,0xfb},{0x3e,0x00},{0x43,0x11},{0x16,0x10},{0x39,0x92},
    {0x35,0xda},{0x22,0x1a},{0x37,0xc3},{0x23,0x00},{0x34,0xc0},{0x36,0x1a},
    {0x06,0x88},{0x07,0xc0},{0x0d,0x87},{0x0e,0x41},{0x4c,0x00},{0x48,0x00},
    {0x5b,0x00},{0x42,0x03},{0x4a,0x81},{0x21,0x99},{0x24,0x40},{0x25,0x38},
    {0x26,0x82},{0x5c,0x00},{0x63,0x00},{0x61,0x70},{0x62,0x80},{0x7c,0x05},
    {0x20,0x80},{0x28,0x30},{0x6c,0x00},{0x6d,0x80},{0x6e,0x00},{0x70,0x02},
    {0x71,0x94},{0x73,0xc1},{0x12,0x40},{0x17,0x11},{0x18,0x43},{0x19,0x00},
    {0x1a,0x4b},{0x32,0x09},{0x37,0xc0},{0x4f,0x60},{0x50,0xa8},{0x6d,0x00},
    {0x3d,0x38},{0x46,0x3f},{0x4f,0x60},{0x0c,0x3c},{0xff,0x00},{0xe5,0x7f},
    {0xf9,0xc0},{0x41,0x24},{0xe0,0x14},{0x76,0xff},{0x33,0xa0},{0x42,0x20},
    {0x43,0x18},{0x4c,0x00},{0x87,0xd5},{0x88,0x3f},{0xd7,0x03},{0xd9,0x10},
    {0xd3,0x82},{0xc8,0x08},{0xc9,0x80},{0x7c,0x00},{0x7d,0x00},{0x7c,0x03},
    {0x7d,0x48},{0x7d,0x48},{0x7c,0x08},{0x7d,0x20},{0x7d,0x10},{0x7d,0x0e},
    {0x90,0x00},{0x91,0x0e},{0x91,0x1a},{0x91,0x31},{0x91,0x5a},{0x91,0x69},
    {0x91,0x75},{0x91,0x7e},{0x91,0x88},{0x91,0x8f},{0x91,0x96},{0x91,0xa3},
    {0x91,0xaf},{0x91,0xc4},{0x91,0xd7},{0x91,0xe8},{0x91,0x20},{0x92,0x00},
    {0x93,0x06},{0x93,0xe3},{0x93,0x05},{0x93,0x05},{0x93,0x00},{0x93,0x04},
    {0x93,0x00},{0x93,0x00},{0x93,0x00},{0x93,0x00},{0x93,0x00},{0x93,0x00},
    {0x93,0x00},{0x96,0x00},{0x97,0x08},{0x97,0x19},{0x97,0x02},{0x97,0x0c},
    {0x97,0x24},{0x97,0x30},{0x97,0x28},{0x97,0x26},{0x97,0x02},{0x97,0x98},
    {0x97,0x80},{0x97,0x00},{0x97,0x00},{0xc3,0xed},{0xa4,0x00},{0xa8,0x00},
    {0xc5,0x11},{0xc6,0x51},{0xbf,0x80},{0xc7,0x10},{0xb6,0x66},{0xb8,0xa5},
    {0xb7,0x64},{0xb9,0x7c},{0xb3,0xaf},{0xb4,0x97},{0xb5,0xff},{0xb0,0xc5},
    {0xb1,0x94},{0xb2,0x0f},{0xc4,0x5c},{0xc0,0x64},{0xc1,0x4b},{0x8c,0x00},
    {0x86,0x3d},{0x50,0x00},{0x51,0xc8},{0x52,0x96},{0x53,0x00},{0x54,0x00},
    {0x55,0x00},{0x5a,0xc8},{0x5b,0x96},{0x5c,0x00},{0xd3,0x00},{0xc3,0xed},
    {0x7f,0x00},{0xda,0x00},{0xe5,0x1f},{0xe1,0x67},{0xe0,0x00},{0xdd,0x7f},
    {0x05,0x00},{0x12,0x40},{0xd3,0x04},{0xc0,0x16},{0xc1,0x12},{0x8c,0x00},
    {0x86,0x3d},{0x50,0x00},{0x51,0x2c},{0x52,0x24},{0x53,0x00},{0x54,0x00},
    {0x55,0x00},{0x5a,0x2c},{0x5b,0x24},{0x5c,0x00},{0xff,0xff}
};

static const uint8_t OV2640_320x240_JPEG[][2] = {
    {0xff,0x01},{0x12,0x40},{0x17,0x11},{0x18,0x43},{0x19,0x00},{0x1a,0x4b},
    {0x32,0x09},{0x4f,0xca},{0x50,0xa8},{0x5a,0x23},{0x6d,0x00},{0x39,0x12},
    {0x35,0xda},{0x22,0x1a},{0x37,0xc3},{0x23,0x00},{0x34,0xc0},{0x36,0x1a},
    {0x06,0x88},{0x07,0xc0},{0x0d,0x87},{0x0e,0x41},{0x4c,0x00},{0xff,0x00},
    {0xe0,0x04},{0xc0,0x64},{0xc1,0x4b},{0x86,0x35},{0x50,0x89},{0x51,0xc8},
    {0x52,0x96},{0x53,0x00},{0x54,0x00},{0x55,0x00},{0x57,0x00},{0x5a,0x50},
    {0x5b,0x3c},{0x5c,0x00},{0xe0,0x00},{0xff,0xff}
};

/* ============================================================
 * 12. SD PROTOCOL
 * ============================================================ */
static uint8_t sd_cmd(uint8_t cmd, uint32_t arg)
{
    uint8_t crc = 0xFF;
    if (cmd == CMD0) crc = 0x95;
    if (cmd == CMD8) crc = 0x87;

    spi_rx(sd_spi);
    spi_xfer(sd_spi, 0x40 | cmd);
    spi_xfer(sd_spi, (arg >> 24) & 0xFF);
    spi_xfer(sd_spi, (arg >> 16) & 0xFF);
    spi_xfer(sd_spi, (arg >>  8) & 0xFF);
    spi_xfer(sd_spi,  arg        & 0xFF);
    spi_xfer(sd_spi, crc);

    uint8_t r = 0xFF;
    for (int i = 0; i < 8; i++) {
        r = spi_rx(sd_spi);
        if (!(r & 0x80)) return r;
    }
    return 0xFF;
}

static uint8_t sd_acmd(uint8_t acmd, uint32_t arg)
{
    sd_cmd(CMD55, 0);
    return sd_cmd(acmd, arg);
}

/* Read OCR via CMD58 and print it. Returns OCR word, or 0 on failure.
 * Bit 31 = card power-up done. If this never sets, the card isn't getting
 * enough voltage/current to finish its internal init. */
static uint32_t sd_read_ocr(void)
{
    uint8_t r1 = sd_cmd(CMD58, 0);
    if (r1 > R1_IDLE) return 0;
    uint8_t o[4];
    for (int i = 0; i < 4; i++) o[i] = spi_rx(sd_spi);
    return ((uint32_t)o[0] << 24) | ((uint32_t)o[1] << 16) |
           ((uint32_t)o[2] <<  8) |  (uint32_t)o[3];
}

/* Full init. Diagnostics + three strategies:
 *   1. ACMD41 with HCS   (standard SDHC, 8 s)
 *   2. ACMD41 without HCS (4 s)
 *   3. CMD1               (MMC-style, 4 s)
 * Returns 0 on success. */
static int sd_init_card(void)
{
    spi_reset(sd_spi);

    spi_cs(sd_spi, 0);
    for (int i = 0; i < 10; i++) spi_rx(sd_spi);   /* 80 clocks, CS high */

    spi_cs(sd_spi, 1);
    uint8_t r1 = 0xFF;
    for (int i = 0; i < 10; i++) {
        r1 = sd_cmd(CMD0, 0);
        if (r1 == R1_IDLE) break;
        usleep(1000);
    }
    if (r1 != R1_IDLE) {
        printf("  [sd] CMD0 failed (0x%02X)\n", r1);
        spi_cs(sd_spi, 0);
        return -1;
    }

    int v2_card = 0;
    r1 = sd_cmd(CMD8, 0x000001AA);
    if (r1 == R1_IDLE) {
        uint8_t e[4];
        for (int i = 0; i < 4; i++) e[i] = spi_rx(sd_spi);
        printf("  [sd] CMD8 R7: %02X %02X %02X %02X ", e[0], e[1], e[2], e[3]);
        if (e[2] != 0x01 || e[3] != 0xAA) {
            printf("(echo MISMATCH)\n");
            spi_cs(sd_spi, 0);
            return -2;
        }
        printf("(ok, v2 card)\n");
        v2_card = 1;
    } else {
        printf("  [sd] CMD8 r1=0x%02X (v1/MMC card path)\n", r1);
    }

    /* OCR snapshot before init — shows voltage window + busy bit */
    uint32_t ocr = sd_read_ocr();
    printf("  [sd] OCR before init: 0x%08lX (busy bit31=%lu)\n",
           (unsigned long)ocr, (unsigned long)((ocr >> 31) & 1));

    /* --- Strategy 1: ACMD41 with HCS, up to 8 s --- */
    for (int i = 0; i < 8000; i++) {
        r1 = sd_acmd(ACMD41, v2_card ? 0x40000000 : 0);
        if (r1 == R1_READY) goto ready;
        usleep(1000);
        if (i % 2000 == 1999) {
            ocr = sd_read_ocr();
            printf("  [sd] still idle after %ds, OCR=0x%08lX\n",
                   (i + 1) / 1000, (unsigned long)ocr);
        }
    }
    printf("  [sd] ACMD41 (HCS) stalled. Trying ACMD41 without HCS...\n");

    /* --- Strategy 2: ACMD41 without HCS, 4 s --- */
    for (int i = 0; i < 4000; i++) {
        r1 = sd_acmd(ACMD41, 0);
        if (r1 == R1_READY) goto ready;
        usleep(1000);
    }
    printf("  [sd] ACMD41 (no HCS) stalled. Trying CMD1...\n");

    /* --- Strategy 3: CMD1 (old SD / MMC), 4 s --- */
    for (int i = 0; i < 4000; i++) {
        r1 = sd_cmd(CMD1, 0);
        if (r1 == R1_READY) goto ready;
        usleep(1000);
    }

    ocr = sd_read_ocr();
    printf("  [sd] INIT FAILED. Final OCR=0x%08lX\n", (unsigned long)ocr);
    printf("  [sd] Card answers commands but never finishes internal\n");
    printf("  [sd] power-up. With comms proven good, this is almost\n");
    printf("  [sd] always UNDERVOLTAGE: the module needs 4.5-5.5V on\n");
    printf("  [sd] VCC (it has an onboard LDO). Pmod VCC pins are 3.3V!\n");
    printf("  [sd] -> Move module VCC to a 5V pin, keep signals on JC.\n");
    spi_cs(sd_spi, 0);
    return -3;

ready:
    /* Determine addressing mode */
    if (v2_card) {
        ocr = sd_read_ocr();
        sd_is_sdhc = (ocr & 0x40000000) ? 1 : 0;
        printf("  [sd] Init OK. OCR=0x%08lX\n", (unsigned long)ocr);
    } else {
        sd_is_sdhc = 0;
        printf("  [sd] Init OK (v1/MMC path).\n");
    }
    if (!sd_is_sdhc) sd_cmd(CMD16, SD_BLOCK_LEN);
    spi_cs(sd_spi, 0);
    spi_rx(sd_spi);
    return 0;
}

/* Init with whole-procedure retries (power dips can abort mid-init). */
static int sd_init_retry(int tries)
{
    for (int t = 1; t <= tries; t++) {
        printf("[SD] Init attempt %d/%d...\n", t, tries);
        if (sd_init_card() == 0) return 0;
        usleep(300000);   /* let the rail recover */
    }
    return -1;
}

/* Quick liveness probe: CMD58 should return R1=0x00 on an initialised card. */
static int sd_probe(void)
{
    spi_cs(sd_spi, 1);
    uint8_t r1 = sd_cmd(CMD58, 0);
    if (r1 == R1_READY) {
        uint8_t ocr[4];
        for (int i = 0; i < 4; i++) ocr[i] = spi_rx(sd_spi);
        spi_cs(sd_spi, 0);
        spi_rx(sd_spi);
        return 0;
    }
    spi_cs(sd_spi, 0);
    spi_rx(sd_spi);
    return -1;
}

static uint32_t sd_blkaddr(uint32_t blk)
{
    return sd_is_sdhc ? blk : blk * SD_BLOCK_LEN;
}

static int sd_write_block(uint32_t blk, const uint8_t *buf)
{
    spi_cs(sd_spi, 1);
    uint8_t r1 = sd_cmd(CMD24, sd_blkaddr(blk));
    if (r1 != R1_READY) {
        spi_cs(sd_spi, 0);
        printf("  [sd] CMD24 rejected (0x%02X) block %u\n", r1, blk);
        return -1;
    }
    spi_rx(sd_spi);
    spi_xfer(sd_spi, DATA_TOKEN);
    for (int i = 0; i < SD_BLOCK_LEN; i++) spi_xfer(sd_spi, buf[i]);
    spi_xfer(sd_spi, 0xFF);
    spi_xfer(sd_spi, 0xFF);
    uint8_t resp = spi_rx(sd_spi);
    if ((resp & 0x1F) != DATA_ACCEPT) {
        spi_cs(sd_spi, 0);
        printf("  [sd] write rejected token=0x%02X block %u\n", resp, blk);
        return -2;
    }
    /* busy-wait while the card programs flash (usleep matches proven code) */
    for (int i = 0; i < 1000000; i++) {
        if (spi_rx(sd_spi) == 0xFF) break;
        usleep(10);
    }
    spi_cs(sd_spi, 0);
    spi_rx(sd_spi);
    return 0;
}

/* ============================================================
 * 13. PHASE 0 — SD INIT FIRST (lowest board load)
 * ============================================================ */
static int phase_sd_init(void)
{
    printf("\n=== PHASE 0: SD INIT (SPI0 @ 0x%08lX) — camera still asleep ===\n",
           (unsigned long)SD_SPI_BASE);
    if (sd_init_retry(3) != 0) {
        printf("[SD] Init FAILED even before camera was started.\n");
        printf("     → This is now a pure power/card problem, not interference.\n");
        return -1;
    }
    printf("[SD] Card ready. Type: %s\n", sd_is_sdhc ? "SDHC/SDXC" : "SDSC");
    return 0;
}

/* ============================================================
 * 14. PHASE 1 — CAMERA
 * ============================================================ */
static int phase_camera(void)
{
    printf("\n=== PHASE 1: CAMERA (SPI1 @ 0x%08lX) ===\n",
           (unsigned long)CAM_SPI_BASE);

    cam_spi_init();

    /* Make sure sensor is awake (clear PWDN in case of a previous run) */
    cam_write_reg(ARDUCHIP_GPIO, 0x00);
    usleep(20000);

    printf("[CAM] SPI test... ");
    cam_write_reg(ARDUCHIP_TEST1, 0x55);
    uint8_t t = cam_read_reg(ARDUCHIP_TEST1);
    if (t != 0x55) {
        printf("FAIL (got 0x%02X)\n", t);
        return -1;
    }
    printf("PASS\n");

    printf("[CAM] Sensor init (OV2640 320x240)... ");
    i2c_write(0xFF, 0x01);
    i2c_write(0x12, 0x80);
    usleep(200000);
    i2c_write_array(OV2640_JPEG_INIT);
    i2c_write_array(OV2640_320x240_JPEG);
    printf("DONE\n");

    printf("[CAM] Triggering capture...\n");
    cam_write_reg(ARDUCHIP_FIFO, 0x01);
    cam_write_reg(ARDUCHIP_FIFO, 0x02);

    int attempts = 0;
    while (!(cam_read_reg(ARDUCHIP_TRIG) & CAP_DONE_MASK)) {
        usleep(100000);
        printf("."); fflush(stdout);
        if (++attempts > 500) { printf("\n[CAM] TIMEOUT\n"); return -2; }
    }
    printf("\n[CAM] Capture done.\n");

    uint32_t s1 = cam_read_reg(ARDUCHIP_FIFO_SZ1);
    uint32_t s2 = cam_read_reg(ARDUCHIP_FIFO_SZ2);
    uint32_t s3 = cam_read_reg(ARDUCHIP_FIFO_SZ3) & 0x7F;
    uint32_t fifo_len = ((s3 << 16) | (s2 << 8) | s1) & 0x07FFFFF;
    printf("[CAM] FIFO size: %u bytes\n", fifo_len);

    uint32_t read_len = fifo_len;
    if (read_len == 0 || read_len > IMG_BYTES + 8) {
        printf("[CAM] Suspicious size — clamping to %d bytes\n", IMG_BYTES);
        read_len = IMG_BYTES;
    }
    if (read_len > IMG_BYTES) read_len = IMG_BYTES;

    printf("[CAM] Reading %u bytes from FIFO...\n", read_len);
    memset(image_buf, 0, IMG_BYTES);

    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFE;
    cam_xfer(BURST_FIFO_READ);
    cam_xfer(0x00);
    for (uint32_t i = 0; i < read_len; i++) {
        image_buf[i] = cam_xfer(0x00);
        if (i % 8192 == 0) { printf("."); fflush(stdout); }
    }
    *(volatile uint32_t *)((uint8_t *)cam_spi + XSPI_SSR) = 0xFFFFFFFF;
    printf("\n[CAM] Read complete. %u bytes in buffer.\n", read_len);

    /* Save image.bin on the board rootfs right away */
    FILE *f = fopen("/image.bin", "wb");
    if (f) {
        fwrite(image_buf, 1, IMG_BYTES, f);
        fclose(f);
        printf("[CAM] Saved /image.bin (%d bytes) — scp-able now.\n", IMG_BYTES);
    } else {
        printf("[CAM] WARNING: couldn't write /image.bin (%s)\n", strerror(errno));
    }

    /* Sensor really goes to sleep now */
    cam_powerdown();
    usleep(50000);
    return 0;
}

/* ============================================================
 * 15. PHASE 2 — SD WRITE
 * ============================================================ */
static int phase_sd_write(void)
{
    printf("\n=== PHASE 2: SD WRITE ===\n");

    /* Card was initialised in phase 0. Check it's still alive. */
    printf("[SD] Probing card (CMD58)... ");
    if (sd_probe() == 0) {
        printf("alive, no re-init needed.\n");
    } else {
        printf("lost state — re-initialising (camera now asleep)...\n");
        if (sd_init_retry(3) != 0) {
            printf("[SD] Re-init FAILED.\n");
            return -1;
        }
    }

    printf("[SD] Writing %d bytes (%d blocks) starting at block %u...\n",
           IMG_BYTES, IMG_BLOCKS, IMAGE_START_BLOCK);

    uint8_t hdr[SD_BLOCK_LEN];
    memset(hdr, 0, SD_BLOCK_LEN);
    snprintf((char *)hdr, SD_BLOCK_LEN,
             "CAM_IMG W=%d H=%d BYTES=%d BLOCKS=%d FMT=YUV422",
             IMG_W, IMG_H, IMG_BYTES, IMG_BLOCKS);
    printf("[SD] Header at block %u... ", IMAGE_START_BLOCK - 1);
    printf("%s\n", sd_write_block(IMAGE_START_BLOCK - 1, hdr) == 0
                   ? "OK" : "FAILED (continuing)");

    int errors = 0;
    for (int b = 0; b < IMG_BLOCKS; b++) {
        if (sd_write_block(IMAGE_START_BLOCK + (uint32_t)b,
                           image_buf + (uint32_t)b * SD_BLOCK_LEN) != 0)
            errors++;
        if (b % 30 == 0) { printf("."); fflush(stdout); }
    }
    printf("\n[SD] Written %d/%d blocks. Errors: %d\n",
           IMG_BLOCKS - errors, IMG_BLOCKS, errors);

    spi_cs(sd_spi, 0);
    return errors ? -1 : 0;
}

/* ============================================================
 * 16. MAIN
 * ============================================================ */
int main(void)
{
    printf("==============================================\n");
    printf("  cam_sd v3: SD-first, sensor PWDN, OCR diagnostics\n");
    printf("  Image: %dx%d YUV422 (%d bytes, %d blocks)\n",
           IMG_W, IMG_H, IMG_BYTES, IMG_BLOCKS);
    printf("  SD blocks: %u (header) + %u..%u (data)\n",
           IMAGE_START_BLOCK - 1, IMAGE_START_BLOCK,
           IMAGE_START_BLOCK + IMG_BLOCKS - 1);
    printf("==============================================\n");

    printf("[INIT] Mapping peripherals...\n");
    cam_spi  = map_periph(CAM_SPI_BASE);
    sd_spi   = map_periph(SD_SPI_BASE);
    i2c_base = map_periph(I2C_BASE);

    if (phase_sd_init() != 0) {
        printf("[FATAL] SD never initialised — fix power/card before camera tests.\n");
        return 1;
    }

    if (phase_camera() != 0) {
        printf("[FATAL] Camera phase failed.\n");
        return 1;
    }

    if (phase_sd_write() != 0) {
        printf("[FATAL] SD write phase failed.\n");
        printf("        (Image is still safe at /image.bin — scp it.)\n");
        return 1;
    }

    printf("\n[SUCCESS] Image captured and written to SD card.\n");
    printf("  Option A: scp root@<board_ip>:/image.bin .   (direct)\n");
    printf("  Option B: ./read_sd  → image_from_sd.bin, then scp that\n");
    printf("  Then run your existing Python recovery script.\n");
    return 0;
}
