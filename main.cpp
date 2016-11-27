#include <unistd.h>
#include "header.h"

extern "C"
void multi_gpus(std::string cur_dir, unsigned int mem_divider);

unsigned long int get_pixels_dir(std::string src_dir);

int main(int argc, char **argv){
	if(argc != 2){
		std::cout<<"Command line should have one parameter to indicate memory divider"<<std::endl;
		return 0;
	}
	
	unsigned int mem_divider = std::stoi(argv[1]);

	// get current working directory
	char cwd[MAX_DIR_LEN];
	if(getcwd(cwd, sizeof(cwd)) == NULL)
		std::cout<<"Get image source directory FAILED!!!"<<std::endl;

	std::string cur_dir(cwd);
	multi_gpus(cur_dir, mem_divider);
	return 0;
}

unsigned long int get_pixels_dir(std::string cur_dir){
	std::string src_dir_str(cur_dir+"/src");
	std::cout<<"Image source: "<<src_dir_str<<std::endl;
	std::string img_src_ent_str;

	// open the directory of source and enumarate the image files in the directory
	DIR *img_src = opendir(src_dir_str.c_str());

	// if source and target folder open failed
	if(img_src == NULL){
		std::cout<<"Can not to open the directory!!!"<<std::endl;
		return 0;
	}

	unsigned int total_pixel_cnt = 0;
	dirent *img_ent;
	cv::Mat img;
	unsigned int img_cnt = 0;
	// enumerate every image file in current directory and process
	while((img_ent = readdir(img_src))){
		if(strcmp(img_ent->d_name, ".") == 0 || strcmp(img_ent->d_name, "..")==0)
			continue;
		else{
			img_cnt++;
			img_src_ent_str = src_dir_str + "/" + img_ent->d_name;
			img = cv::imread(img_src_ent_str, CV_LOAD_IMAGE_COLOR);
			if(img.empty())
				continue;
			total_pixel_cnt += img.total();
			img_src_ent_str.clear();
		}
	}
	std::cout<<"There are "<<img_cnt<<" images"<<std::endl<<std::endl;
	return total_pixel_cnt;
}
