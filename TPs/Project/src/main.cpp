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
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	GLFWwindow* window = glfwCreateWindow(1024, 1024, "ISIMA_PROJECT", nullptr, nullptr);
	if (window) {
		glfwMakeContextCurrent(window);
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glDisable(GL_DEPTH_TEST);

		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGLSetGLDevice(0);
		cudaGraphicsResource_t cuda_graphics_resource;
		cudaGraphicsGLRegisterImage(&cuda_graphics_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
		GLuint fbo = 0;
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		glfwSwapInterval(1);
		glfwSetKeyCallback(window, key_callback);

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			//interop
			cudaGraphicsMapResources(1, &cuda_graphics_resource);
			cudaArray_t writeArray;
			cudaGraphicsSubResourceGetMappedArray(&writeArray, cuda_graphics_resource, 0, 0);
			cudaResourceDesc wdsc;
			wdsc.resType = cudaResourceTypeArray;
			wdsc.res.array.array = writeArray;
			cudaSurfaceObject_t surface;
			cudaCreateSurfaceObject(&surface, &wdsc);
			//TODO Draw the map on the surface
			cudaDestroySurfaceObject(surface);
			cudaGraphicsUnmapResources(1, &cuda_graphics_resource);
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
