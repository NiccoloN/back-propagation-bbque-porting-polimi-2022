/**
 * @author Niccolò Nicolosi <niccolo.nicolosi@mail.polimi.it>
 */

#ifndef BACKPROP_PORTING_EXC_H_
#define BACKPROP_PORTING_EXC_H_

#include <bbque/bbque_exc.h>
#include <iostream>
#include <chrono>
#include <dlb.h>
#include <dlb_drom.h>

extern "C" {

#include "omp.h"
#include "backprop.h"
}

using bbque::rtlib::BbqueEXC;
using namespace std;

class BackpropPorting_exc : public BbqueEXC {

public:

    BackpropPorting_exc(std::string const & name,
		    std::string const & recipe,
		    RTLIB_Services_t *rtlib,
            int argc,
            char *argv[]);

private:

	RTLIB_ExitCode_t onSetup();
	RTLIB_ExitCode_t onConfigure(int8_t awm_id);
	RTLIB_ExitCode_t onRun();
	RTLIB_ExitCode_t onMonitor();
	RTLIB_ExitCode_t onSuspend();
	RTLIB_ExitCode_t onRelease();

private:

    int layer_size;
    BPNN *net;
    int in, hid, out;
    float out_err, hid_err;

    int work_step;
    int counter;

    int32_t proc_nr;    // nr. of CPU cores
    int32_t proc_quota; // CPU quota (e.g. quota = 235% -> 3 CPU cores)
    int32_t acc, gpu;

    int pid;
};

#endif // BACKPROP_PORTING_EXC_H_
