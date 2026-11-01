/* MLKit's POSIX seek wrapper uses C int offsets. Keep GGUF file offsets
 * 64-bit even for models larger than 1 GiB; parsing stays in Standard ML. */
#define _FILE_OFFSET_BITS 64
#include <stdint.h>
#include <unistd.h>
int64_t fuchat_seek(int64_t fd, int64_t offset) {
  return (int64_t)lseek((int)fd, (off_t)offset, SEEK_SET);
}
