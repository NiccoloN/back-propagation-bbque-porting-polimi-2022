
# Compilation

	$ mkdir build
	$ cd build
	$ cmake .. -DBOSP_SYSROOT=path-to-bosp/out/usr -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=path-to-bosp/out/usr
	$ make
	# make install # necessary to install the .recipe file

# Execution

- Start the BOSP daemon
	[BOSPShell >] bbque-startd

- Launch the application
	/opt/bosp/usr/bin/backprop-porting n

- Stop the BOSP daemon
	[BOSPShell >] bbque-stopd
