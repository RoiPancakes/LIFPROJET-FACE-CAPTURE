solution "gKit2light"
	configurations { "debug", "release" }

	platforms { "x64", "x32" }

	includedirs { ".", "src/gKit", "lib" }

	gkit_dir = path.getabsolute(".")


	configuration "debug"
		targetdir "bin/debug"
		defines { "DEBUG" }
		flags { "Symbols" }

	configuration "release"
		targetdir "bin/release"
--~ 		defines { "NDEBUG" }
--~ 		defines { "GK_RELEASE" }
		flags { "OptimizeSpeed" }

	configuration "linux"
		buildoptions { "-mtune=native -march=native" }
		buildoptions { "-std=c++11 -mavx -O3" }
		buildoptions { "-W -Wall -Wextra -Wsign-compare -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable", "-pipe" }
		buildoptions { "-flto"}
		linkoptions { "-flto -lpthread -lX11 -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core"}
		links { "GLEW", "SDL2", "SDL2_image", "GL" }

	configuration { "linux", "debug" }
		buildoptions { "-g"}
		linkoptions { "-g"}

	configuration { "linux", "release" }
		buildoptions { "-fopenmp" }
		linkoptions { "-fopenmp" }

	configuration { "windows" }
		defines { "WIN32", "NVWIDGETS_EXPORTS", "_USE_MATH_DEFINES", "_CRT_SECURE_NO_WARNINGS" }
		defines { "NOMINMAX" } -- allow std::min() and std::max() in vc++ :(((

	configuration { "windows", "gmake", "x32" }
		buildoptions { "-std=c++11"}
		includedirs { "extern/mingw/include" }
		libdirs { "extern/mingw/lib" }
		links { "mingw32", "SDL2main", "SDL2", "SDL2_image", "opengl32", "glew32" }

	configuration { "windows", "codeblocks", "x32" }
		buildoptions { "-std=c++11"}
		includedirs { "extern/mingw/include" }
		libdirs { "extern/mingw/lib" }
		links { "mingw32", "SDL2main", "SDL2", "SDL2_image", "opengl32", "glew32" }

	configuration { "windows", "vs2013" }
		if _PREMAKE_VERSION >="5.0" then
			system "Windows"
			architecture "x64"
			disablewarnings { "4244", "4305" }
		end
		includedirs { "extern/visual2013/include" }
		libdirs { "extern/visual2013/lib" }
		platforms { "x64" }
		links { "opengl32", "glew32", "SDL2", "SDL2main", "SDL2_image" }


	configuration { "windows", "vs2015" }
		if _PREMAKE_VERSION >="5.0" then
			system "Windows"
			architecture "x64"
			disablewarnings { "4244", "4305" }
			flags { "MultiProcessorCompile", "NoMinimalRebuild" }
		end
		includedirs { "extern/visual2015/include" }
		libdirs { "extern/visual2015/lib" }
		links { "opengl32", "glew32", "SDL2", "SDL2main", "SDL2_image" }

	configuration { "windows", "vs2017" }
		if _PREMAKE_VERSION >="5.0" then
			system "Windows"
			architecture "x64"
			disablewarnings { "4244", "4305" }
			flags { "MultiProcessorCompile", "NoMinimalRebuild" }
		end

		-- memes librairies que pour la version 2015
		includedirs { "extern/visual2015/include" }
		libdirs { "extern/visual2015/lib" }
		links { "opengl32", "glew32", "SDL2", "SDL2main", "SDL2_image" }

	configuration "macosx"
		frameworks= "-F /Library/Frameworks/"
		buildoptions { "-std=c++11" }
		defines { "GK_MACOS" }
		buildoptions { frameworks }
		linkoptions { frameworks .. " -framework OpenGL -framework SDL2 -framework SDL2_image" }


 -- description des fichiers communs
gkit_files = { gkit_dir .. "/src/gKit/*.cpp", gkit_dir .. "/src/gKit/*.h" }


-- quand ce premake4.lua est inclus par un autre premake qui definit no_project=true (donc quand gkit2light est utilisé comme une lib),
-- ceci stoppe la creation des projects suivants (tuto, etc.)
if no_project then
	do return end
end


project("FaceCapture")
	language "C++"
	kind "ConsoleApp"
	targetdir "bin"
	files ( gkit_files )
	files { gkit_dir .. "/src/FaceCapture/main.cpp" }
	-- include de la DLIB
	files { gkit_dir .. "/lib/dlib/all/source.cpp" }
