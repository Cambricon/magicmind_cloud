#include <gflags/gflags.h>
#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include "net_params.h"

DEFINE_string(network, "BODY_25", "network name, BODY_25 and COCO are supported.");

const NetParams &GetNetParams() {
  static const std::unordered_map<std::string, NetParams> kparams_map {
    std::make_pair("BODY_25", NetParams({
      25,  // nkeypoints, ignore background
      26,  // nbody_part_pairs
      78,  // nheatmaps
      std::vector<cv::Point>({  // body_part_pairs
        cv::Point(1, 8), {1, 2}, {1, 5}, {2, 3}, {3, 4},
	{5 ,  6}, {6 ,  7}, {8 ,  9}, {9 , 10}, {10, 11},
	{8 , 12}, {12, 13}, {13, 14}, {1 ,  0}, {0 , 15},
	{15, 17}, {0 , 16}, {16, 18}, {2 , 17}, {5 , 18},
	{14, 19}, {19, 20}, {14, 21}, {11, 22}, {22, 23},
	{11, 24}
      }),
      std::vector<cv::Point>({  // paf_indexs
        cv::Point(0, 1), {14, 15}, {22, 23}, {16, 17}, {18, 19},
	{24, 25}, {26, 27}, {6 ,  7}, {2 ,  3}, {4 ,  5},
	{8 ,  9}, {10, 11}, {12, 13}, {30, 31}, {32, 33},
	{36, 37}, {34, 35}, {38, 39}, {20, 21}, {28, 29},
	{40, 41}, {42, 43}, {44, 45}, {46, 47}, {48, 49},
	{50, 51}
      }),
      //  render params
      std::vector<cv::Scalar>({  // colors
	cv::Scalar(255,     0,    85), 
        cv::Scalar(255,     0,     0), 
        cv::Scalar(255,    85,     0), 
        cv::Scalar(255,   170,     0), 
        cv::Scalar(255,   255,     0), 
        cv::Scalar(170,   255,     0), 
        cv::Scalar( 85,   255,     0), 
        cv::Scalar(  0,   255,     0), 
        cv::Scalar(255,     0,     0), 
        cv::Scalar(  0,   255,    85), 
        cv::Scalar(  0,   255,   170), 
        cv::Scalar(  0,   255,   255), 
        cv::Scalar(  0,   170,   255), 
        cv::Scalar(  0,    85,   255), 
        cv::Scalar(  0,     0,   255), 
        cv::Scalar(255,     0,   170), 
        cv::Scalar(170,     0,   255), 
        cv::Scalar(255,     0,   255), 
        cv::Scalar( 85,     0,   255), 
        cv::Scalar(  0,     0,   255), 
        cv::Scalar(  0,     0,   255), 
        cv::Scalar(  0,     0,   255), 
        cv::Scalar(  0,   255,   255), 
        cv::Scalar(  0,   255,   255), 
        cv::Scalar(  0,   255,   255)
      }),
      std::vector<cv::Point>({  // render_body_part_pairs
        cv::Point(1, 8), {1, 2}, {1, 5}, {2, 3}, {3, 4},
	{5 ,  6}, {6 ,  7}, {8 ,  9}, {9 , 10}, {10, 11},
	{8 , 12}, {12, 13}, {13, 14}, {1 ,  0}, {0 , 15},
	{15, 17}, {0 , 16}, {16, 18}, {14, 19}, {19, 20},
	{14, 21}, {11, 22}, {22, 23}, {11, 24}
      }),
      std::vector<int>({  // indexes_in_coco_order
        0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11
      })
    })),
    std::make_pair("COCO", NetParams({
      18,  // nkeypoints, ignore background, one more keypoint than COCO dataset(neck)
      19,  // nbody_part_pairs
      57,  // nheatmaps
      std::vector<cv::Point>({  // body_part_pairs
        cv::Point(1, 2), {1, 5}, {2, 3}, {3, 4}, {5, 6},
        {6 ,  7}, {1 ,  8}, {8 ,  9}, {9 , 10}, {1 , 11},
        {11, 12}, {12, 13}, {1 ,  0}, {0 , 14}, {14, 16},
        {0 , 15}, {15, 17}, {2 , 16}, {5 , 17}
      }),
      std::vector<cv::Point>({  // paf_indexs
        cv::Point(12, 13), {20, 21}, {14, 15}, {16, 17}, {22, 23},
	{24, 25}, {0 ,  1}, {2 ,  3}, {4 ,  5}, {6 ,  7},
	{8 ,  9}, {10, 11}, {28, 29}, {30, 31}, {34, 35},
	{32, 33}, {36, 37}, {18, 19}, {26, 27}
      }),
      //  render params
      std::vector<cv::Scalar>({  // colors
        cv::Scalar(255,     0,    85),
        cv::Scalar(255,     0,     0),
        cv::Scalar(255,    85,     0),
        cv::Scalar(255,   170,     0),
        cv::Scalar(255,   255,     0),
        cv::Scalar(170,   255,     0),
        cv::Scalar( 85,   255,     0),
        cv::Scalar(  0,   255,     0),
        cv::Scalar(  0,   255,    85),
        cv::Scalar(  0,   255,   170),
        cv::Scalar(  0,   255,   255),
        cv::Scalar(  0,   170,   255),
        cv::Scalar(  0,    85,   255),
        cv::Scalar(  0,     0,   255),
        cv::Scalar(255,     0,   170),
        cv::Scalar(170,     0,   255),
        cv::Scalar(255,     0,   255),
        cv::Scalar( 85,     0,   255)
      }),
      std::vector<cv::Point>({  // render_body_part_pairs
        cv::Point(1, 2), {1, 5}, {2, 3}, {3, 4}, {5, 6},
	{6 ,  7}, {1 ,  8}, {8 ,  9}, {9 , 10}, {1 , 11},
	{11, 12}, {12, 13}, {1 ,  0}, {0 , 14}, {14, 16},
	{0 , 15}, {15, 17}
      }),
      std::vector<int>({  // indexes_in_coco_order
        0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10
      })
    }))
  };

  auto iter = kparams_map.find(FLAGS_network);
  LOG_IF(FATAL, kparams_map.end() == iter)
    << "Can not find network[" << FLAGS_network << "] implementation.";
  return iter->second;
}

const std::string &GetNetName() {
  return FLAGS_network;
}

