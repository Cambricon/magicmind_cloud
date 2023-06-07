#include "post_process.hpp"
#include "utils.hpp"
std::map<int, std::string> load_name(std::string name_map_file) {
  std::map<int, std::string> name_map;
  std::ifstream in(name_map_file);
  if (!in) {
    std::cout << "failed to load imagenet_name file: " + name_map_file + ".\n";
    exit(0);
  }
  std::string line;
  int index = 0;
  std::string name;
  while (getline(in, line)) {
    int found = line.find(' ');
    name = line.substr(found + 1);
    name_map[index] = name;
    index += 1;
  }
  return name_map;
}

std::vector<int> ArgTopK(const float *data, int classes, int k) {
  std::vector<int> result;
  result.reserve(k);
  for (int i = 0; i < k; ++i)
    result.push_back(i);
  auto comp = [data](int a, int b) { return data[a] > data[b]; };
  std::make_heap(result.begin(), result.end(), comp);
  for (int i = 0; i < classes; ++i) {
    if (comp(i, result[0])) {
      std::pop_heap(result.begin(), result.end(), comp);
      result.back() = i;
      std::push_heap(result.begin(), result.end(), comp);
    }
  }
  std::sort(result.begin(), result.end(), comp);
  return result;
}
