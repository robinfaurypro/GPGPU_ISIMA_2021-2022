#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <gpgpu.h>

constexpr int32_t kWidth = 1024;
constexpr int32_t kHeight = 1024;

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

void main() {
	if (!glfwInit()) {
		glfwTerminate();
	}
	//Select the OpenGL version 4.1
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	GLFWwindow* window = glfwCreateWindow(1024, 1024, "ISIMA_PROJECT", nullptr, nullptr);
	if (window) {
		//Set the OpenGL context available. OpenGL function can be called after this function
		glfwMakeContextCurrent(window);
		//Load the OpenGL function pointer from the graphic library.
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

		//Clear the background to black
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glDisable(GL_DEPTH_TEST);

		//Creation of the OpenGL texture
		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGLSetGLDevice(0);
		cudaGraphicsResource_t cuda_graphic_resource;
		cudaGraphicsGLRegisterImage(&cuda_graphic_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		//OpenGL object to print the texture on screen
		GLuint fbo = 0;
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		//Limit the FPS to 60 (0 to set to unlimited)
		glfwSwapInterval(1);

		glfwSetKeyCallback(window, key_callback);

		//Creation of the cuda resource desc
		cudaResourceDesc cuda_resource_desc;
		memset(&cuda_resource_desc, 0, sizeof(cuda_resource_desc));
		cuda_resource_desc.resType = cudaResourceTypeArray;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		
		//Creation of a surface cuda
		cudaArray_t array;
		cudaMallocArray(&array, &channelDesc, kWidth, kHeight, cudaArraySurfaceLoadStore);
		cuda_resource_desc.res.array.array = array;
		cudaSurfaceObject_t surface = 0;
		cudaCreateSurfaceObject(&surface, &cuda_resource_desc);

		//Main loop
		while (!glfwWindowShouldClose(window)) {
			showFPS(window);
			glfwPollEvents();
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			//Update the simulation for each frame
			//TODO

			//Setup the OpenGL texture 
			cudaGraphicsMapResources(1, &cuda_graphic_resource);
			cudaArray_t array_OpenGL;
			cudaGraphicsSubResourceGetMappedArray(&array_OpenGL, cuda_graphic_resource, 0, 0);
			cuda_resource_desc.res.array.array = array_OpenGL;
			cudaSurfaceObject_t surface_OpenGL;
			cudaCreateSurfaceObject(&surface_OpenGL, &cuda_resource_desc);
			//Copy the cuda surface into the surface_OpenGL
			//CopyTo(surface, surface_OpenGL, kWidth, kHeight);
			cudaDestroySurfaceObject(surface_OpenGL);
			cudaGraphicsUnmapResources(1, &cuda_graphic_resource);
			cudaStreamSynchronize(0);

			glViewport(0, 0, width, height);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
			glBlitFramebuffer(
				0, 0, kWidth, kHeight,
				0, 0, width, height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glfwSwapBuffers(window);
		}
		glfwDestroyWindow(window);
	}
	glfwTerminate();
}
