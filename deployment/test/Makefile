LIB = /usr/local/lib/*.so
MACRO = -D SPEED_TEST
CUR_DIR = $(shell pwd)
DEFAULT_MODEL_PATH = $(CUR_DIR)/model.onnx

Test_ClassifierModule: Test_ClassifierModule.cpp
	g++ -o Test_ClassifierModule.out Test_ClassifierModule.cpp $(LIB) $(MACRO)

.PHONY: run
run: Test_ClassifierModule
	./Test_ClassifierModule.out $(DEFAULT_MODEL_PATH)
.PHONY: clean
clean:
	rm -rf *.out

