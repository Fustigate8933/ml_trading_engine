#pragma once

#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>

class ITCHReader {
private:
    uint8_t *start_p;
    uint8_t *p;
    size_t data_size;
    int fd;
    std::filesystem::path data_path;

public:
    ITCHReader(const std::string &path) : data_path{path} {
        fd = open(data_path.c_str(), O_RDONLY);

        if (fd == -1) {
            throw std::runtime_error("Failed to open file");
        }

        struct stat sb;

        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Error getting file size");
        }

        data_size = sb.st_size;

        start_p = reinterpret_cast<uint8_t *>(mmap(nullptr, data_size, PROT_READ, MAP_PRIVATE, fd, 0));

        if (start_p == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }

        madvise(start_p, data_size, MADV_SEQUENTIAL);

        p = start_p;
    }

    ~ITCHReader() {
        munmap(start_p, data_size);
        close(fd);
    }

    uint8_t *next() {
        if (start_p + data_size <= p + 2) {
            return nullptr;
        }

        uint16_t msg_len;
        std::memcpy(&msg_len, p, 2);
        msg_len = __builtin_bswap16(msg_len);
        p += 2;

        uint8_t *msg_p = p;
        p += msg_len;

        return msg_p;
    }
};
