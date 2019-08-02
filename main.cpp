
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#define MAX_SOURCE_SIZE (0x100000)


int main(void) {
	printf("started running\n");

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	cv::Mat img_test, imgRGBA, img_copyRGBA;
	img_test = cv::imread("test.jpg");
	cv::cvtColor(img_test, imgRGBA, CV_BGR2RGBA);


	//cv::imshow("test", img_test);

	cv::Mat img_copy(img_test.rows, img_test.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::cvtColor(img_copy, img_copyRGBA, CV_BGR2RGBA);

	fp = fopen("copy_image.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	printf("kernel loading done\n");

	// Get platform and device information
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;


	cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(ret_num_platforms*sizeof(cl_platform_id));

	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
	printf("ret at %d is %d\n", __LINE__, ret);

	ret = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 1,
		&device_id, &ret_num_devices);
	printf("ret at %d is %d\n", __LINE__, ret);
	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Image format
	cl_image_format format;
	format.image_channel_order = CL_RGBA;
	format.image_channel_data_type = CL_UNORM_INT8;

	// Image desc
	cl_image_desc desc;
	desc.image_height = img_test.rows;
	desc.image_width = img_test.cols;
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_depth = 0;
	desc.image_array_size = 0;
	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;

	// Img
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { img_test.rows, img_test.cols, 1 };

	// Error code ret
	cl_int errcode_ret;
	printf("%d", CL_INVALID_IMAGE_DESCRIPTOR);

	char *buffer = reinterpret_cast<char *> (&img_test.data);

	// Create memory buffers on the device for each vector 
	cl_mem in_buffer_obj = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		&format, &desc, (void*)imgRGBA.data, &errcode_ret);

	cl_mem out_buffer_obj = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
		&format, &desc, (void*)img_copyRGBA.data, &errcode_ret);

	//ret = clEnqueueReadImage(command_queue, out_buffer_obj, CL_TRUE, origin, region, 0, 0, (void*)img_copy.data, 0, NULL, NULL);


	printf("ret at %d is %d\n", __LINE__, ret);

	printf("ret at %d is %d\n", __LINE__, ret);

	printf("before building\n");
	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char **)&source_str, (const size_t *)&source_size, &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	printf("ret at %d is %d\n", __LINE__, ret);

	printf("after building\n");
	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "copy_image", &ret);
	printf("ret at %d is %d\n", __LINE__, ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_buffer_obj);
	printf("ret at %d is %d\n", __LINE__, ret);

	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_buffer_obj);
	printf("ret at %d is %d\n", __LINE__, ret);


	printf("before execution\n");
	// Execute the OpenCL kernel on the list
	size_t global_item_size[2] = {img_test.rows, img_test.cols}; // Process the entire image
	size_t local_item_size[2] = {1, 1}; // Divide work items into groups of 64
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		global_item_size, local_item_size, 0, NULL, NULL);
	printf("after execution\n");

	printf("%d", CL_INVALID_COMMAND_QUEUE);
	printf("%d", CL_INVALID_MEM_OBJECT);

	
	ret = clEnqueueReadImage(command_queue, out_buffer_obj, CL_TRUE, origin, region, 0, 0, (void*)img_copyRGBA.data, 0, NULL, NULL);
	
	printf("after copying\n");
	// Display the result to the screen
	cv::imshow("result", img_copyRGBA);
	cv::waitKey(0);

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(in_buffer_obj);
	ret = clReleaseMemObject(out_buffer_obj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	return 0;
}