// roce_proxy.c's interface where libibverbs' headers are missing at build time (decode D1): the layout as there, and
// every open refused, so cuda_comm.zig keeps NCCL for every gather.
#include <stdint.h>
#include <stdio.h>

#define ROCE_SLOTS 2
#define ROCE_PORTS_MAX 2
#define ROCE_FLAG_STRIDE 128

typedef struct tf_roce tf_roce;

uint64_t tf_roce_blob_bytes(void) { return 72; }

int tf_roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
  if (world < 2 || world > 2 || slot_bytes == 0 || slot_bytes % 4096 != 0 || slot_bytes > (1ull << 30)) return -1;
  out[0] = 0;
  out[1] = (uint64_t)world * ROCE_SLOTS * slot_bytes;
  out[2] = out[1] + (uint64_t)world * ROCE_SLOTS * ROCE_PORTS_MAX * ROCE_FLAG_STRIDE;
  out[3] = out[2] + ROCE_SLOTS * slot_bytes;
  out[4] = out[3] + ROCE_FLAG_STRIDE;
  return 0;
}

tf_roce *tf_roce_open(int world, int rank, const char *const *names, int ports, int gid_index, void *region,
                      uint64_t region_bytes, uint64_t slot_bytes, char *err, uint64_t err_len) {
  (void)world, (void)rank, (void)names, (void)ports, (void)gid_index, (void)region, (void)region_bytes, (void)slot_bytes;
  snprintf(err, err_len, "built without libibverbs headers: no RoCE all-gather");
  return NULL;
}
int tf_roce_blob_of(tf_roce *r, void *out) { (void)r, (void)out; return -1; }
int tf_roce_connect(tf_roce *r, const void *peer) { (void)r, (void)peer; return -1; }
int tf_roce_start(tf_roce *r) { (void)r; return -1; }
int tf_roce_failed(tf_roce *r) { (void)r; return 1; }
const char *tf_roce_error(tf_roce *r) { (void)r; return "no RoCE in this build"; }
uint64_t tf_roce_ops(tf_roce *r) { (void)r; return 0; }
void tf_roce_close(tf_roce *r) { (void)r; }
