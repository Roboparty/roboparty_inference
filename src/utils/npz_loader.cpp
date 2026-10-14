// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#include "npz_loader.hpp"

#include <zlib.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
std::vector<unsigned char> read_bytes(std::ifstream& input, size_t size) {
    if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("NPZ entry exceeds addressable stream size");
    std::vector<unsigned char> bytes(size);
    if (size && !input.read(reinterpret_cast<char*>(bytes.data()), size))
        throw std::runtime_error("Truncated NPZ archive");
    return bytes;
}

uint64_t read_le(const std::vector<unsigned char>& bytes, size_t offset, size_t width) {
    if (offset > bytes.size() || width > bytes.size() - offset)
        throw std::runtime_error("Truncated ZIP header field");
    uint64_t value = 0;
    for (size_t i = 0; i < width; ++i) value |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
    return value;
}

struct EntrySizes {
    uint64_t compressed;
    uint64_t uncompressed;
};

EntrySizes entry_sizes(const std::vector<unsigned char>& header,
                       const std::vector<unsigned char>& extra) {
    EntrySizes sizes{read_le(header, 18, 4), read_le(header, 22, 4)};
    const bool need_compressed = sizes.compressed == 0xffffffffU;
    const bool need_uncompressed = sizes.uncompressed == 0xffffffffU;
    if (!need_compressed && !need_uncompressed) return sizes;

    // ZIP64 stores uncompressed size first, then compressed size, for sentinel fields only.
    for (size_t offset = 0; offset < extra.size();) {
        const uint64_t tag = read_le(extra, offset, 2);
        const size_t length = read_le(extra, offset + 2, 2);
        offset += 4;
        if (length > extra.size() - offset) throw std::runtime_error("Truncated ZIP extra field");
        if (tag == 0x0001) {
            const size_t required = 8 * (need_uncompressed + need_compressed);
            if (length < required) throw std::runtime_error("Truncated ZIP64 size field");
            if (need_uncompressed) sizes.uncompressed = read_le(extra, offset, 8);
            if (need_compressed) sizes.compressed = read_le(extra, offset + (need_uncompressed ? 8 : 0), 8);
            return sizes;
        }
        offset += length;
    }
    throw std::runtime_error("Missing ZIP64 size field");
}

std::vector<unsigned char> decode_entry(std::ifstream& input, const EntrySizes& sizes,
                                       uint64_t method, uint32_t expected_crc) {
    if (sizes.compressed > std::numeric_limits<uInt>::max() ||
        sizes.uncompressed > std::numeric_limits<uInt>::max() ||
        sizes.compressed > std::numeric_limits<size_t>::max() ||
        sizes.uncompressed > std::numeric_limits<size_t>::max())
        throw std::runtime_error("NPZ entry exceeds the zlib single-buffer size limit");
    auto bytes = read_bytes(input, static_cast<size_t>(sizes.compressed));
    if (method == 0) {
        if (sizes.compressed != sizes.uncompressed)
            throw std::runtime_error("Stored NPZ entry size mismatch");
    } else {
        std::vector<unsigned char> decoded(static_cast<size_t>(sizes.uncompressed));
        z_stream stream{};
        stream.next_in = bytes.data();
        stream.avail_in = static_cast<uInt>(bytes.size());
        stream.next_out = decoded.data();
        stream.avail_out = static_cast<uInt>(decoded.size());
        if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
            throw std::runtime_error("Cannot initialize NPZ decompressor");
        const int status = inflate(&stream, Z_FINISH);
        const bool complete = status == Z_STREAM_END && stream.total_in == sizes.compressed &&
                              stream.total_out == sizes.uncompressed;
        inflateEnd(&stream);
        if (!complete) throw std::runtime_error("Corrupt NPZ compressed data or size mismatch");
        bytes = std::move(decoded);
    }
    uLong checksum = crc32(0, Z_NULL, 0);
    for (size_t offset = 0; offset < bytes.size();) {
        const auto count = static_cast<uInt>(std::min<size_t>(bytes.size() - offset, 1 << 20));
        checksum = crc32(checksum, bytes.data() + offset, count);
        offset += count;
    }
    if (checksum != expected_crc) throw std::runtime_error("NPZ entry checksum mismatch");
    return bytes;
}

cnpy::NpyArray parse_array(std::vector<unsigned char>& bytes) {
    // cnpy's buffer parser supports NPY version 1 headers.
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0 ||
        bytes[6] != 1 || bytes[7] != 0)
        throw std::runtime_error("Expected an NPY version 1 array");
    const size_t header_size = bytes[8] | (static_cast<size_t>(bytes[9]) << 8);
    const size_t data_offset = 10 + header_size;
    if (header_size == 0 || data_offset > bytes.size() || bytes[data_offset - 1] != '\n')
        throw std::runtime_error("Invalid NPY header length");
    size_t word_size;
    std::vector<size_t> shape;
    bool fortran_order;
    cnpy::parse_npy_header(bytes.data(), word_size, shape, fortran_order);
    size_t payload_size = word_size;
    for (size_t dim : shape) {
        if (dim != 0 && payload_size > std::numeric_limits<size_t>::max() / dim)
            throw std::runtime_error("NPY array size overflow");
        payload_size *= dim;
    }
    if (word_size == 0 || payload_size != bytes.size() - data_offset)
        throw std::runtime_error("NPY payload size mismatch");
    cnpy::NpyArray result(shape, word_size, fortran_order);
    if (payload_size) std::memcpy(result.data<unsigned char>(), bytes.data() + data_offset, payload_size);
    return result;
}

cnpy::npz_t load(const std::string& path, const std::string* wanted) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open NPZ archive: " + path);
    const auto end = input.tellg();
    if (end < 22) throw std::runtime_error("Truncated NPZ archive");
    const auto file_size = static_cast<uint64_t>(end);
    input.seekg(0);
    cnpy::npz_t arrays;
    while (true) {
        auto header = read_bytes(input, 4);
        const uint64_t signature = read_le(header, 0, 4);
        if (signature == 0x02014b50 || signature == 0x06054b50) break;
        if (signature != 0x04034b50) throw std::runtime_error("Invalid NPZ local header");
        const auto rest = read_bytes(input, 26);
        header.insert(header.end(), rest.begin(), rest.end());
        const uint64_t flags = read_le(header, 6, 2);
        if (flags & 0x0001) throw std::runtime_error("Encrypted NPZ entries are not supported");
        if (flags & 0x0008) throw std::runtime_error("NPZ data descriptors are not supported");
        const uint64_t method = read_le(header, 8, 2);
        if (method != 0 && method != 8) throw std::runtime_error("Unsupported NPZ compression method");
        const auto filename = read_bytes(input, read_le(header, 26, 2));
        const auto extra = read_bytes(input, read_le(header, 28, 2));
        const auto sizes = entry_sizes(header, extra);
        const auto position = input.tellg();
        if (position < 0 || static_cast<uint64_t>(position) > file_size ||
            sizes.compressed > file_size - static_cast<uint64_t>(position))
            throw std::runtime_error("Truncated NPZ entry");
        std::string name(filename.begin(), filename.end());
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".npy") == 0) {
            name.resize(name.size() - 4);
            if (!wanted || name == *wanted) {
                const auto checksum = static_cast<uint32_t>(read_le(header, 14, 4));
                auto bytes = decode_entry(input, sizes, method, checksum);
                if (!arrays.emplace(name, parse_array(bytes)).second)
                    throw std::runtime_error("Duplicate NPZ array: " + name);
                if (wanted) return arrays;
                continue;
            }
        }
        // Skip the compressed size, which can differ from the decoded NPY size.
        input.seekg(static_cast<std::streamoff>(sizes.compressed), std::ios::cur);
        if (!input) throw std::runtime_error("Cannot skip NPZ entry");
    }
    if (wanted) throw std::runtime_error("NPZ array not found: " + *wanted);
    return arrays;
}
}

cnpy::npz_t roboparty::load_npz(const std::string& path) { return load(path, nullptr); }
cnpy::NpyArray roboparty::load_npz(const std::string& path, const std::string& name) {
    return load(path, &name).at(name);
}
