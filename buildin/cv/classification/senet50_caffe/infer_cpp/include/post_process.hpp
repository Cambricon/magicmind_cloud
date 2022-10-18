#ifndef POST_PROCESS_HPP
#define POST_PROCESS_HPP

std::map<int, std::string> load_name(std::string name_map_file);
std::vector<int> ArgTopK(const float *data, int classes, int k);
#endif // POST_PROCESS_HPP

