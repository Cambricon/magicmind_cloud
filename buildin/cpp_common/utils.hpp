#ifndef _SAMPLE_UTILS111_HPP
#define _SAMPLE_UTILS111_HPP

#include <fstream>
#include <iostream>
#include "sys/stat.h"
#include <mm_runtime.h>
#include <cnrt.h>

#include <mm_runtime.h>

class Record {
 public:
  Record(std::string filename) {
    outfile.open((filename).c_str(), std::ios::trunc | std::ios::out);
  }

  ~Record() {
    if (outfile.is_open())
      outfile.close();
  }

  void write(std::string line, bool print = false) {
    outfile << line << std::endl;
    if (print) {
      std::cout << line << std::endl;
    }
  }

 private:
  std::ofstream outfile;
};

inline bool check_file_exist(std::string path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0) {
    if ((buffer.st_mode & S_IFDIR) == 0)
      return true;
    return false;
  }
  return false;
}

inline bool check_folder_exist(std::string path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == 0) {
    if ((buffer.st_mode & S_IFDIR) == 0)
      return false;
    return true;
  }
  return false;
}

void PrintModelInfo(magicmind::IModel *model);
#endif  // _SAMPLE_UTILS111_HPP
