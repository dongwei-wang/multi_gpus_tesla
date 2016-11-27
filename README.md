# brief introduction
I do images pixels conversion from RGB to gray scales
All images are stored at src/ directory which I specifiec at program
It can process any size of images only if you have enough big hard drives
Just store the images into src/, this frame work will process them

I implemented a frame with CUDA programming model with multiple GPU devices in ace-tesla.egr.vcu.edu server,
I only employ the six tesla GPUs to process image pixels 
In each GPU device, I also divide the data into multi streams

# Create folder to store the source color image and target grayscale images
To make sure the program could run, we should create three folders in project directory:

src/ : store the colored images

cpu_tar/ : store the grayscale images which are processed by CPU

gpu_tar/ : store the grayscale images which are processed by GPU

# Run the program

Command line parameter: 
In multiple GPU implementation, we should input a number at command line, this parameter indicate how do we divide the memory since I observed that my program will break down if I allocate too many memory size, this server is not solely available for me, we share it, we have to increase this parameter to allocate less memory size for each processing.

Compile and run:

put the images into src/ directory

	make

	sh run

program will work


