/**
 * @author Niccolò Nicolosi <niccolo.nicolosi@mail.polimi.it>
 */

#include <libgen.h>
#include <iostream>
#include <memory>

#include "BackpropPorting_exc.h"

using namespace std;

int main(int argc, char *argv[]) 
{
	RTLIB_Services_t *rtlib;
	auto ret = RTLIB_Init(basename(argv[0]), &rtlib);
	if (ret != RTLIB_OK) {
		cerr << "ERROR: Unable to init RTLib (Did you start the BarbequeRTRM daemon?)" << endl;
		return RTLIB_ERROR;
	}
	assert(rtlib);

	std::string recipe("aem-template");
	cout << "INFO: Registering EXC with recipe " << recipe << endl;
	auto pexc = std::make_shared<BackpropPorting_exc>("BackpropPorting_exc", recipe, rtlib, argc, argv);
	if (!pexc->isRegistered()) {
		cerr << "ERROR: Register failed (missing the recipe file?)" << endl;
		return RTLIB_ERROR;
	}

	cout << "INFO: Starting EXC control thread " << endl;
	pexc->Start();

	cout << "INFO: Waiting for the EXC termination " << endl;
	pexc->WaitCompletion();

	cout << "INFO: Terminated. " << endl;
	return EXIT_SUCCESS;
}

