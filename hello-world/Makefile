
all: install compile run

install-dep:;
	sudo apt-get install cmake default-jre python3-pip imx500-all imx500-tools
	pip3 install opencv-python-headless --break-system-packages
	pip3 install torch --break-system-packages
	pip3 install torchvision --break-system-packages
	pip3 install onnx --break-system-packages
	pip3 install datasets --break-system-packages
	pip3 install safetensors --break-system-packages
	pip3 install imx500-converter[pt] --break-system-packages
	pip3 install model_compression_toolkit --break-system-packages
	pip3 uninstall opencv-python opencv-contrib-python opencv-python-headless --break-system-packages
	pip3 install opencv-python-headless --break-system-packages

compile: clean
	python3 compile-to-onnx.py
	imxconv-pt -i model.onnx -o modelbin
	imx500-package -i modelbin/packerOut.zip -o modelpack

run::
	python3 imx500_run_hello_world.py

clean:;
	rm -r -f modelbin modelpack model.onnx

distclean: clean
	rm -r -f __pycache__ *~ 
