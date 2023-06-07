/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description: Some function for file read and data process.
 *************************************************************************/
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include "data.h"
std::string AddLocalPathIfName(const std::string &filepath) {
  if (!filepath.empty()) {
    std::string::size_type pos = filepath.find("/");
    if (pos == std::string::npos) {
      return "./" + filepath;
    }
  }
  return filepath;
}

std::vector<std::string> StringSplit(const std::string &s, const std::string &delimiter) {
  std::vector<std::string> result;
  std::string::size_type pos1, pos2;
  pos2 = s.find(delimiter);
  pos1 = 0;
  while (std::string::npos != pos2) {
    result.push_back(s.substr(pos1, pos2 - pos1));
    pos1 = pos2 + delimiter.size();
    pos2 = s.find(delimiter, pos1);
  }
  if (pos1 != s.length()) {
    result.push_back(s.substr(pos1));
  }
  return result;
}

bool CreateFolder(const std::string &file_path) {
  std::string cur_dir = "";
  std::vector<std::string> dirname_vec = StringSplit(file_path, "/");
  for (auto &dirname : dirname_vec) {
    cur_dir = cur_dir + dirname + "/";
    if (access(cur_dir.c_str(), R_OK) != 0) {
      if (mkdir(cur_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) != 0) {
        return false;
      }
    }
  }
  return true;
}

size_t FileSize(const std::string &file_path) {
  std::ifstream in_file(file_path, std::ios::in | std::ios::binary);
  CHECK_VALID(in_file);
  // get length of file:
  in_file.seekg(0, in_file.end);
  size_t length = in_file.tellg();
  in_file.close();
  return length;
}

bool ReadDataFromFile(const std::string &file_path, void *ptr, size_t size) {
  std::ifstream in_file(file_path, std::ios::in | std::ios::binary);
  if (!in_file) {
    SLOG(ERROR) << "Open file " << file_path << " failed during read.";
    return false;
  }
  if (!ptr) {
    SLOG(ERROR) << "Ptr invalid for " << file_path << " during read.";
    in_file.close();
    return false;
  }
  // get length of file:
  in_file.seekg(0, in_file.end);
  size_t length = in_file.tellg();
  in_file.seekg(0, in_file.beg);
  if (size > length) {
    SLOG(ERROR) << "Request for " << size << " size of data but " << file_path << " length is "
                << length << ".";
    in_file.close();
    return false;
  }
  in_file.read((char *)ptr, size);
  if (!in_file) {
    SLOG(ERROR) << "Request for " << size << " size of data but only " << in_file.gcount()
                << " can be read in " << file_path << ".";
    in_file.close();
    return false;
  }
  in_file.close();
  return true;
}

bool WriteDataToFile(const std::string &file_path, void *ptr, size_t size) {
  std::ofstream out_file(file_path, std::ios::out | std::ios::binary);
  if (!out_file) {
    SLOG(ERROR) << "Open file " << file_path << " failed during write.";
    return false;
  }
  if (!ptr) {
    SLOG(ERROR) << "Ptr invalid for " << file_path << " during write.";
    out_file.close();
    return false;
  }
  out_file.write((char *)ptr, size);
  out_file.close();
  return true;
}

bool ReadListFromFile(const std::string &file_path,
                      std::vector<std::string> *lines,
                      std::vector<std::vector<int>> *shapes) {
  std::ifstream in_file(file_path, std::ios::in | std::ios::binary);
  if (!in_file) {
    SLOG(ERROR) << "Open file " << file_path << " failed during read.";
    return false;
  }
  if (!lines || !shapes) {
    SLOG(ERROR) << "Ptr invalid for " << file_path << " during read.";
    in_file.close();
    return false;
  }
  bool bad_parse = false;
  lines->clear();
  shapes->clear();
  std::string line{};
  std::vector<std::string> tokens;
  while (getline(in_file, line)) {
    tokens.clear();
    std::istringstream iss(line);
    std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
              std::back_inserter(tokens));
    if (tokens.size() == 1) {
      if (!line.empty()) {
        lines->push_back(line);
      }
    } else if (tokens.size() == 2) {
      lines->push_back(tokens[0]);
      if ((tokens[1].rfind("shape[", 0) == 0) && (tokens[1].back() == ']')) {
        auto nums = tokens[1].substr(6, tokens[1].size() - 7);
        std::vector<int> shape;
        std::string num;
        std::stringstream shape_ss(nums);
        while (getline(shape_ss, num, ',')) {
          shape.push_back(std::stoi(num));
        }
        shapes->push_back(shape);
        continue;
      }
      bad_parse = true;
    } else {
      bad_parse = true;
    }
  }
  if (bad_parse) {
    SLOG(ERROR) << "File " << file_path << " bad format.";
  }
  in_file.close();
  return !bad_parse;
}

bool ReadLabelFromFile(const std::string &file_path,
                       std::vector<std::string> *images,
                       std::vector<int> *labels) {
  std::ifstream flabel(file_path, std::ios::in);
  if (!flabel) {
    SLOG(ERROR) << "Open file " << file_path << " failed during read.";
    return false;
  }
  if (!images || !labels) {
    SLOG(ERROR) << "Ptr invalid for " << file_path << " during read.";
    flabel.close();
    return false;
  }
  images->clear();
  labels->clear();
  std::string line{};
  std::vector<std::string> tokens;
  const std::vector<std::string> support_exts = {".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".GIF"};
  while (getline(flabel, line)) {
    tokens.clear();
    std::istringstream iss(line);
    std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
              std::back_inserter(tokens));
    if (tokens.size() != 2) {
      SLOG(ERROR) << "File " << file_path
                  << " format can not be splited into img name and its label.";
      flabel.close();
      return false;
    }
    bool done = false;
    for (auto e_ : support_exts) {
      auto file_name = tokens[0];
      if (file_name.size() < e_.size()) {
        continue;
      } else {
        file_name = file_name.substr(file_name.size() - e_.size());
      }
      std::transform(file_name.begin(), file_name.end(), file_name.begin(), ::toupper);
      if (file_name != e_) {
        continue;
      } else {
        done = true;
        break;
      }
    }
    if (!done) {
      SLOG(WARNING) << "File " << file_path << " contains unsupport filename " << tokens[0];
      SLOG(WARNING) << "It may cause a failure of loading image.";
    }
    images->push_back(tokens[0]);
    labels->push_back(std::stoi(tokens[1]));
  }
  flabel.close();
  return true;
}
