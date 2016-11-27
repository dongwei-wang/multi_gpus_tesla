#include "header.h"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

extern "C"
void multi_gpus(std::string cur_dir, unsigned int mem_divider);

__global__ void multi_gpus_kernel(unsigned char* input, unsigned char* output, int stream_idx, unsigned int pixels_per_stream ,unsigned int pixel_cnt) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned idx = stream_idx * pixels_per_stream + tid;

	if( tid < pixel_cnt ){
		unsigned int color_idx = idx * BYTES_PER_PIXEL;
		unsigned int gray_idx = idx;

		unsigned char blue	= input[color_idx];
		unsigned char green	= input[color_idx + 1];
		unsigned char red	= input[color_idx + 2];
		float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		output[gray_idx] = static_cast<unsigned char>(gray);
	}
}


// get free memory space
unsigned long int get_free_mem(){
	std::string token;
	std::ifstream file("/proc/meminfo");
	while(file >> token){
		if(token == "MemFree:"){
			unsigned long int free_mem;
			if(file >> free_mem)
				return free_mem;
		}
	}
	return 0;
}

void gray_processing_CPU(unsigned char* color_pixels, unsigned char* gray_pixels, unsigned long int pixel_cnt){
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	unsigned char blue;
	unsigned char green;
	unsigned char red;
	float gray;
	std::cout<<"CPU processing......"<<std::endl;
	for( unsigned long int i=0; i<pixel_cnt; i++ ){
		blue	= color_pixels[i*3];
		green	= color_pixels[i*3+1];
		red	= color_pixels[i*3+2];
		gray	= red * 0.3f + green * 0.59f + blue * 0.11f;
		gray_pixels[i] = static_cast<unsigned char>(gray);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"CPU time: "<<milliseconds<< " ms"<<std::endl;
}

void gray_processing_multi_streams(unsigned char* color_pixels, unsigned char* gray_pixels, unsigned long int pixel_cnt){
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	unsigned char *d_pixels_in, *d_pixels_out;
	SAFE_CALL(cudaMalloc(&d_pixels_in, pixel_cnt * BYTES_PER_PIXEL * sizeof( unsigned char )), "Malloc colored device memory failed!");
	SAFE_CALL(cudaMalloc(&d_pixels_out, pixel_cnt * sizeof(unsigned char)), "Malloc grayed device memory failed!");

	// get the number of pixesl each stream
	unsigned pixels_per_stream = (pixel_cnt % STREAMS_CNT == 0 )? ( pixel_cnt/STREAMS_CNT ) : ( pixel_cnt/STREAMS_CNT + 1 );
	dim3 block(BLOCK_SIZE);
	dim3 grid((pixels_per_stream + BLOCK_SIZE -1 )/BLOCK_SIZE);

	std::cout<<"GPU processing......"<<std::endl;

	// crate streams for current device
	cudaStream_t* streams = (cudaStream_t*) malloc( STREAMS_CNT * sizeof( cudaStream_t ) );
	for(int i=0; i<STREAMS_CNT; i++)
		SAFE_CALL(cudaStreamCreate(&streams[i]), "Create stream failed!");

	unsigned int pixel_in_cur_stream = 0;

	// start the stream execution for current device
	for( int i=0; i<STREAMS_CNT; i++ ){
		// this is the boundary check for the pixel number in last stream
		// normally, it coule not be the same number of pixels in each stream
		if ( i == STREAMS_CNT -1  )
			pixel_in_cur_stream = pixel_cnt - pixels_per_stream * (STREAMS_CNT - 1);
		else
			pixel_in_cur_stream = pixels_per_stream;

		// copy data from host to device
		SAFE_CALL(cudaMemcpyAsync(&d_pixels_in[i * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof( unsigned char )],
								&color_pixels[i * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof( unsigned char )],
								pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof(unsigned char),
								cudaMemcpyHostToDevice,
								streams[i]),
								"Device memory asynchronized copy failed!");

		// kernel launch
		multi_gpus_kernel<<< grid, block, 0, streams[i] >>>(d_pixels_in, d_pixels_out, i, pixel_in_cur_stream, pixel_cnt);

		// copy data back from device to host
		SAFE_CALL(cudaMemcpyAsync(&gray_pixels[i * pixel_in_cur_stream ],
								&d_pixels_out[i * pixel_in_cur_stream ],
								pixel_in_cur_stream * sizeof(unsigned char),
								cudaMemcpyDeviceToHost,
								streams[i]),
								"Host memory asynchronized copy failed!");
	}

	// synchronize
	cudaDeviceSynchronize();

	// release the memory in device
	SAFE_CALL(cudaFree(d_pixels_in),"Free device color memory failed!");
	SAFE_CALL(cudaFree(d_pixels_out), "Free device gray memory failed!");

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"GPU time: "<<milliseconds<< " ms"<<std::endl;
}

void gray_processing_multi_gpus(unsigned char* color_pixels,
				unsigned char* gray_pixels,
				unsigned long int pixel_cnt,
				int device_cnt){
	//std::cout<<"Total pixel count for this processing: "<<pixel_cnt<<std::endl;
	// time statistics
	float milliseconds[device_cnt];

	// the ratio of memory for each device
	// we allocate the memory size for each device according to these ratio
	//int device_mem_ratio[] = {11,5,2,2,2,2,5,11};
	int device_mem_ratio[] = {2, 1, 1, 1, 1, 2};

	// device_pixel_cnt must be initialized as 0
	unsigned long int device_pixel_cnt[device_cnt];
	for( int i=0; i<device_cnt; i++){
		device_pixel_cnt[i] = 0;
	}

	// allocate pixels to each GPU device according to their free global memory
	//unsigned long int mem_gap = (pixel_cnt + 39)/(11*2+5*2+2*4);
	unsigned long int mem_gap = (pixel_cnt + 7)/8;
	// std::cout<<"Max pixels: "<<max_pixels_cnt<<std::endl;
	//std::cout<<"Mem gap: "<<mem_gap<<std::endl;

	// calculate the pixel counts in each GPU device
	unsigned long int total_pixels = 0;
	for( int i=0; i<device_cnt; i++ ){
		if(total_pixels + device_mem_ratio[i]*mem_gap > pixel_cnt){
			device_pixel_cnt[i] = pixel_cnt - total_pixels;
		} else{
			device_pixel_cnt[i] = (device_mem_ratio[i])*mem_gap;
		}
		total_pixels += device_mem_ratio[i]*mem_gap;
	}

//	for( int i=0; i<device_cnt; i++ ){
//		std::cout<<"Device "<<i<<" pixel count: "<<device_pixel_cnt[i]<<" ---> memory size(GBs): "<<(double)(device_pixel_cnt[i]*(BYTES_PER_PIXEL+1))/(double)(1024*1024*1024)<<std::endl;
//	}
//
	std::cout<<"Multiple GPUs processing......"<<std::endl;

	// declare streams for each device
	// each device includes STREAMS_CNT streams
	// STREAMS_CNT is declared as 4 in header.h
	cudaStream_t streams[device_cnt][STREAMS_CNT];

	// declare the start and stop event for each device
	// these two variables are employed to record GPU time
	cudaEvent_t start[device_cnt], stop[device_cnt];

	// first:
	// allocate the memory for each device accordint to the pixels count
	// create streams for each edevice
	// crate start and stop event
	// we should set device before we initialize all these variables
	unsigned char *d_pixels_in[device_cnt], *d_pixels_out[device_cnt];
	for( int i=0; i<device_cnt; i++ ){
		cudaSetDevice(i);
		// allocate color and gray memory in device
		SAFE_CALL(cudaMalloc(&d_pixels_in[i], device_pixel_cnt[i]*BYTES_PER_PIXEL*sizeof(unsigned char)), "Allocate device color memory failed!");
		SAFE_CALL(cudaMalloc(&d_pixels_out[i], device_pixel_cnt[i]*sizeof(unsigned char)), "Allocate device gray memory failed!");

		// create streams for current device
		for( int j=0; j<STREAMS_CNT; j++ ){
			SAFE_CALL(cudaStreamCreate(&streams[i][j]), "Create stream failed!");
		}

		// create start and stop events
		SAFE_CALL(cudaEventCreate(&start[i]), "Create start event failed");
		SAFE_CALL(cudaEventCreate(&stop[i]), "Create stop event failed");
	}

	// global memory index for the start address of memory copy
	unsigned long int global_index = 0;

	// second:
	// set the threads and block dimension
	// start to record the start time
	// Asynchronized allocate memory for each stream
	for( int i=0; i<device_cnt; i++ ){

		cudaSetDevice(i);

		// block and grid dimension
		unsigned pixels_per_stream = (device_pixel_cnt[i] % STREAMS_CNT == 0) ? (device_pixel_cnt[i]/STREAMS_CNT) : (device_pixel_cnt[i]/STREAMS_CNT + 1);
		dim3 block(BLOCK_SIZE);
		dim3 grid((pixels_per_stream + BLOCK_SIZE -1 )/BLOCK_SIZE);
		unsigned int pixel_in_cur_stream = 0;

		// start event record
		SAFE_CALL(cudaEventRecord(start[i]), "Start record failed");

		// start the stream execution for current device
		for( int j=0; j<STREAMS_CNT; j++ ){
			// this is the boundary check for the pixel number in last stream
			// normally, it coule not be the same number of pixels in each stream
			if ( j == STREAMS_CNT -1  )
				pixel_in_cur_stream = device_pixel_cnt[i] - pixels_per_stream * (STREAMS_CNT - 1);
			else
				pixel_in_cur_stream = pixels_per_stream;

			// copy data from host to device
			SAFE_CALL(cudaMemcpyAsync(&d_pixels_in[i][j * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof( unsigned char )],
						&color_pixels[(global_index+j*pixel_in_cur_stream)*BYTES_PER_PIXEL*sizeof(unsigned char)],
						pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof(unsigned char),
						cudaMemcpyHostToDevice,
						streams[i][j]),
						"Device memory asynchronized copy failed!");

			// kernel launch
			multi_gpus_kernel<<< grid, block, 0, streams[i][j] >>>(d_pixels_in[i], d_pixels_out[i], j, pixel_in_cur_stream, device_pixel_cnt[i]);

			// copy data from device to host
			SAFE_CALL(cudaMemcpyAsync(&gray_pixels[global_index + j*pixel_in_cur_stream ],
						&d_pixels_out[i][j*pixel_in_cur_stream],
						pixel_in_cur_stream*sizeof(unsigned char),
						cudaMemcpyDeviceToHost,
						streams[i][j]),
						"Host memory asynchronized copy failed!");
		}
		// update the global index for next device
		global_index += device_pixel_cnt[i];
		// stop event record
		SAFE_CALL(cudaEventRecord(stop[i]), "Stop record failed" );
	}

	// synchronize stop
	for( int i=0; i<device_cnt; i++){
		cudaSetDevice(i);
		SAFE_CALL(cudaEventSynchronize(stop[i]), "Synchronize stop failed");
	}

	// cuda event elpased time
	for( int i=0; i<device_cnt; i++){
		SAFE_CALL(cudaEventElapsedTime(&milliseconds[i], start[i], stop[i] ),"Get elapsed time failed" );
		std::cout<<"GPU "<<i<<" time: "<<milliseconds[i]<<" ms"<<std::endl;
	}


	// destroy cuda streams
	for( int i=0; i<device_cnt; i++){
		for( int j=0; j<STREAMS_CNT; j++){
			cudaStreamDestroy(streams[i][j]);
		}

	}

	// release the memory in device
	for( int i=0; i<device_cnt; i++ ){
		SAFE_CALL(cudaFree(d_pixels_in[i]),"Free device color memory failed!");
		SAFE_CALL(cudaFree(d_pixels_out[i]), "Free device gray memory failed!");
	}
}

bool compare(unsigned char* gray_pixels_from_cpu, unsigned char* gray_pixels_from_gpu, unsigned long int pixels_cnt){
	for(unsigned long int i=0; i<pixels_cnt; i++){
		if(abs(gray_pixels_from_cpu[i] - gray_pixels_from_gpu[i])>1e-2){
			return false;
		}
	}
	return true;
}



// write the images
void write_images(unsigned char* gray_pixels,
				std::string tar_dir,
				std::vector<std::string> name,
				std::vector<unsigned int> size,
				std::vector<int> row,
				std::vector<int> col,
				unsigned int img_cnt){
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	unsigned int img_ent_start_index = 0;
	std::string ent_str;

	std::cout<<"Writing "<< img_cnt <<" images to hard drives......"<<std::endl<<std::endl;
	// write gray pixels to images
	for(unsigned int i=0; i<img_cnt; i++){
		ent_str = tar_dir + "/" + name[i];
		cv::Mat img_mat(row[i], col[i], CV_8UC1, &gray_pixels[img_ent_start_index]);
		cv::imwrite(ent_str, img_mat, compression_params);
		img_ent_start_index += size[i];
		ent_str.clear();
	}
}

// multiple gpus implementattion
void multi_gpus(std::string cur_dir, unsigned int mem_divider){

	if( mem_divider<=4){
		std::cout<<"Wrong command line parameters"<<std::endl;
		return;
	}

	// CPU information
	unsigned long int tot_free_cpu_mem = get_free_mem()*1024;
	std::cout<<std::endl<<"/***** CPU basic information *****/"<<std::endl;
	std::cout<<"Free memory size in CPU(Bs): "<<tot_free_cpu_mem<<std::endl;
	std::cout<<"Free memory size in CPU(Gs): "<<(double)get_free_mem()/(double)(1024*1024)<<std::endl<<std::endl;

	size_t cur_gpu_free_mem;
	size_t cur_gpu_tot_mem;

	// get the number of GPU devices
	int device_cnt;
	SAFE_CALL(cudaGetDeviceCount(&device_cnt), "Get device count failed!");

	// no cuda supported device
	if( device_cnt<=0 ){
		std::cout<<"There is no CUDA support device!"<<std::endl;
		return;
	}

	// output the GPU information
	unsigned long int tot_free_gpu_mem = 0; // total available memory size in all GPU devices
	std::vector<unsigned long int> gpu_free_mem_size_vec; // available memory size for each GPU device
	for( int i=1; i<device_cnt-1; i++ ){
		cudaSetDevice(i);
		cudaDeviceProp prop;
		SAFE_CALL(cudaGetDeviceProperties(&prop, i), "Get device properity failed!");
		std::cout<<"/***** GPU device basic information *****/"<<std::endl;
		std::cout<<"Devicd index: "<<i<<std::endl;
		std::cout<<"Device name: "<<prop.name<<std::endl;
		std::cout<<"Global memory(Bytes): "<<prop.totalGlobalMem<<std::endl;
		std::cout<<"Global memory(GBs): "<<(double)(prop.totalGlobalMem)/(double)(1024*1024*1024)<<std::endl;
		SAFE_CALL(cudaMemGetInfo(&cur_gpu_free_mem, &cur_gpu_tot_mem), "Get GPU memory info failed!");
		std::cout<<"Free memory(Bytes): "<<cur_gpu_free_mem<<std::endl;
		std::cout<<"Free memory(Gs): "<<(double)cur_gpu_free_mem/(double)(1024*1024*1024)<<std::endl<<std::endl;
		gpu_free_mem_size_vec.push_back(cur_gpu_free_mem);
		tot_free_gpu_mem += cur_gpu_free_mem;
	}

	// output the total memory information of all GPU devices
	std::cout<<"/***** GPU memory information *****/"<<std::endl;
	std::cout<<"Total free memory in GPU(Bytes): "<<tot_free_gpu_mem<<std::endl;
	std::cout<<"Total free memory in GPU(GBs): "<<(double)tot_free_gpu_mem/(double)(1024*1024*1024)<<std::endl<<std::endl;

	//unsigned long int max_mem = (tot_free_cpu_mem<=tot_free_gpu_mem)? tot_free_cpu_mem : tot_free_gpu_mem;
	unsigned long int max_mem = tot_free_gpu_mem;

	// since each pixel take three bytes, we have to divide the max_mem by some number to get the max pixels we can allocate for one process
	unsigned long int max_pixels_cnt = max_mem/mem_divider;
	std::cout<<"Maximul Pixels count: "<< max_pixels_cnt<<std::endl;
	std::cout<<"Maximul memory size: "<< (double)(max_mem)/(double)(1024*1024*1024)<<std::endl<<std::endl;

	cudaSetDevice(0);
	std::string src_dir_str(cur_dir+"/src");
	std::string tar_dir_gpu_str(cur_dir+"/gpu_tar");
	std::string tar_dir_cpu_str(cur_dir+"/cpu_tar");

	// string to store and target the directory of source image entity
	std::string img_src_ent_str, img_tar_cpu_ent_str;
	// open the directory of source and enumarate the image files in the directory
	DIR *img_src = opendir(src_dir_str.c_str());
	// open the directory of target of gpu and store the processed image in the directory
	DIR *img_tar_gpu = opendir(tar_dir_gpu_str.c_str());

	// if source and target folder open failed
	if(img_src == NULL || img_tar_gpu == NULL){
		std::cout<<"Can not to open the directory!!!"<<std::endl;
		return;
	}

	// each image entity in source directory
	dirent *img_ent;
	std::vector<std::string> img_name_vec;
	std::vector<unsigned int> img_size_vec; // store the number of pixels each image
	std::vector<int> img_rows_vec;
	std::vector<int> img_cols_vec;

	std::cout<<"Allocating "<< max_pixels_cnt*(BYTES_PER_PIXEL+1)<< "(Bytes)/"<<(double)(max_pixels_cnt*(BYTES_PER_PIXEL+1))/(double)(1024*1024*1024)<<"(GBs) host memory......"<<std::endl;
	unsigned char *h_color_img_pixels, *h_gray_img_pixels, *gray_img_pixels;
	SAFE_CALL(cudaMallocHost( &h_color_img_pixels, max_pixels_cnt * BYTES_PER_PIXEL * sizeof( unsigned char ) ), "Allocate color host image memory failed!");
	SAFE_CALL(cudaMallocHost( &h_gray_img_pixels, max_pixels_cnt * sizeof( unsigned char )), "Allocate gray host image memory failed!");
	gray_img_pixels = (unsigned char*)malloc( max_pixels_cnt * sizeof(unsigned char) );

	cv::Mat img;
	unsigned int tot_img_cnt = 0;
	unsigned int collect_img_cnt = 0;
	unsigned long int process_pixels_cnt = 0;

	// enumerate every image file in current directory and process
	std::cout<<"Collecting images pixels......"<<std::endl;
	while((img_ent = readdir(img_src))){
		// if detect "." and ".." , we ignore them
		if(strcmp(img_ent->d_name, ".") == 0 || strcmp(img_ent->d_name, "..")==0)
			continue;
		else{
			img_src_ent_str = src_dir_str + "/" + img_ent->d_name;
			img = cv::imread(img_src_ent_str, CV_LOAD_IMAGE_COLOR);

			if(img.empty()){
				std::cout<<"Image Not Found!"<<std::endl;
				return;
			}

			tot_img_cnt++;

			//std::cout<<"Processing "<<tot_img_cnt<<" images"<<std::endl;
			if( process_pixels_cnt + img.total() <= max_pixels_cnt ){
				collect_img_cnt++;
				//std::cout<<"Image name: "<<img_ent->d_name<<std::endl<<std::endl;
				img_name_vec.push_back(img_ent->d_name);
				img_size_vec.push_back(img.total());
				img_rows_vec.push_back(img.rows);
				img_cols_vec.push_back(img.cols);
				memcpy(&h_color_img_pixels[process_pixels_cnt*BYTES_PER_PIXEL*sizeof(unsigned char)],img.ptr(),img.total()*BYTES_PER_PIXEL*sizeof(unsigned char));
				process_pixels_cnt += img.total();
			} else {
				std::cout<<"Process pixels(GBs): "<<(double)(process_pixels_cnt*BYTES_PER_PIXEL)/(double)(1024*1024*1024)<<std::endl<<std::endl;
				gray_processing_CPU(h_color_img_pixels, gray_img_pixels, process_pixels_cnt);
				write_images(gray_img_pixels,tar_dir_cpu_str , img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

				// gpu processing
				// gray_processing_multi_streams(h_color_img_pixels, h_gray_img_pixels, process_pixels_cnt);
				gray_processing_multi_gpus(h_color_img_pixels, h_gray_img_pixels, process_pixels_cnt, device_cnt-2);
				write_images(h_gray_img_pixels, tar_dir_gpu_str, img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

				// clear the variables after one CPU and GPU processing 
				memset(h_color_img_pixels, 0, max_pixels_cnt*BYTES_PER_PIXEL*sizeof(unsigned char)); 
				memset(h_gray_img_pixels, 0, max_pixels_cnt*sizeof(unsigned char));
				memset(gray_img_pixels, 0, max_pixels_cnt*sizeof(unsigned char));
				img_name_vec.clear();
				img_size_vec.clear();
				img_rows_vec.clear();
				img_cols_vec.clear();

				// add current image to variables
				// for next big chunk processing
				collect_img_cnt = 1;
				img_name_vec.push_back(img_ent->d_name);
				img_size_vec.push_back(img.total());
				img_rows_vec.push_back(img.rows);
				img_cols_vec.push_back(img.cols);
				memcpy(&h_color_img_pixels[0], img.ptr(), img.total()*BYTES_PER_PIXEL*sizeof(unsigned char));
				process_pixels_cnt = img.total();
				std::cout<<"Collecting images pixels......"<<std::endl;
			}
		}
	}

	std::cout<<"Process pixels(GBs): "<<(double)(process_pixels_cnt*BYTES_PER_PIXEL)/(double)(1024*1024*1024)<<std::endl<<std::endl;
	gray_processing_CPU(h_color_img_pixels, gray_img_pixels, process_pixels_cnt);
	write_images(gray_img_pixels,tar_dir_cpu_str , img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

	//gray_processing_multi_streams(h_color_img_pixels, h_gray_img_pixels, process_pixels_cnt);
	gray_processing_multi_gpus(h_color_img_pixels, h_gray_img_pixels, process_pixels_cnt, device_cnt-2);
	write_images(h_gray_img_pixels, tar_dir_gpu_str, img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

	// release allocated memory
	free(gray_img_pixels);
	SAFE_CALL(cudaFreeHost(h_color_img_pixels), "Free host color memory failed!");
	SAFE_CALL(cudaFreeHost(h_gray_img_pixels), "Free host gray memory failed!");

	std::cout<<"Images processing completed"<<std::endl;
}
