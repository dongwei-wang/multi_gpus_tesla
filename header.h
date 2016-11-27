#include <iostream>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <dirent.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define BLOCK_SIZE 1024
#define MAX_DIR_LEN 1024
#define BYTES_PER_PIXEL 3
#define STREAMS_CNT 4
