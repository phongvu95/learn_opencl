__kernel void copy_image(__read_only image2d_t srcImg, __write_only image2d_t dstImg)
{
	const sampler_t samplerA = CLK_ADDRESS_NONE;
	
	
	// float4 red = (float4)(255, 255, 255, 1.);
	for(int j=0;j<1300;j++)
	{
		int2 coord = (int2)(get_global_id(0), j);
		
		float4 clr = read_imagef(srcImg, samplerA, coord);
		printf("%d", clr.x)
		float4 red = (float4)(255, 255, 255, 1)
		write_imagef(dstImg, coord, red);
	}
	
}