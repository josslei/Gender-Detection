LIB = /usr/local/lib/*.so
MACRO = -D SPEED_TEST
CUR_DIR = $(shell pwd)
DEFAULT_MODEL_PATH = $(CUR_DIR)/model.onnx
DEFAULT_CAM_INDEX = 0
CXX_COMPILE_FLAG = -O2

Test_ClassifierModule: Test_ClassifierModule.cpp
	g++ -o Test_ClassifierModule.out Test_ClassifierModule.cpp $(LIB) $(MACRO)

Test_libfacedetection: Test_libfacedetection.cpp
	g++ -o Test_libfacedetection.out Test_libfacedetection.cpp $(LIB)

Test_GenderDetection: Test_GenderDetection.cpp
	g++ -o Test_GenderDetection.out Test_GenderDetection.cpp $(LIB)	$(MACRO)

.PHONY: run
run: Test_ClassifierModule Test_libfacedetection
	./Test_ClassifierModule.out $(DEFAULT_MODEL_PATH)
	./Test_libfacedetection.out $(DEFAULT_CAM_INDEX)

.PHONY: clean
clean:
	rm -rf *.out

