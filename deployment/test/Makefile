LIB = /usr/local/lib/*.so
MACRO = -D SPEED_TEST
CUR_DIR = $(shell pwd)
DEFAULT_MODEL_PATH = $(CUR_DIR)/model.onnx
DEFAULT_CAM_INDEX = 0
CXX_COMPILE_FLAG = -O2

Test_ClassifierModule: Test_ClassifierModule.cpp
	g++ -o Test_ClassifierModule Test_ClassifierModule.cpp $(LIB) $(CXX_COMPILE_FLAG) $(MACRO)

Test_libfacedetection: Test_libfacedetection.cpp
	g++ -o Test_libfacedetection Test_libfacedetection.cpp $(LIB) $(CXX_COMPILE_FLAG)

Test_GenderDetection: Test_GenderDetection.cpp
	g++ -o Test_GenderDetection Test_GenderDetection.cpp $(LIB) $(CXX_COMPILE_FLAG)

.PHONY: all
all: Test_ClassifierModule Test_libfacedetection Test_GenderDetection

.PHONY: rebuild
rebuild:
	make clean
	make all

.PHONY: run
run: all
	./Test_ClassifierModule $(DEFAULT_MODEL_PATH)
	./Test_libfacedetection $(DEFAULT_CAM_INDEX)
	./Test_GenderDetection  $(DEFAULT_MODEL_PATH)

.PHONY: clean
clean:
	rm -f Test_ClassifierModule
	rm -f Test_libfacedetection
	rm -f Test_GenderDetection
